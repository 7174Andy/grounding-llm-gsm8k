import torch
import random
import numpy as np

SEED = 0
random.seed(SEED)               
np.random.seed(SEED)            
torch.manual_seed(SEED)         
torch.cuda.manual_seed(SEED) 


import generation_utils
import direction_utils
from control_toolkits import *

import os
import pickle
from tqdm import tqdm
import shutil

TOOLKITS = {
    'rfm' : RFMToolkit,
    'linear' : LinearProbeToolkit,
    'logistic' : LogisticRegressionToolkit,
    'mean_difference' : MeanDifferenceToolkit,
    'pca' : PCAToolkit
}

class NeuralController:
    def __init__(self, model, tokenizer, control_method='rfm', n_components=5, 
                 rfm_iters=10, batch_size=2):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.control_method = control_method
        self.name = None

        print(f"n_components: {n_components}")

        hparams = {
            'control_method' : control_method,
            'rfm_iters' : rfm_iters,
            'forward_batch_size' : batch_size,
            'M_batch_size' : 2048,
            'n_components' : n_components
        }
        self.hyperparams = hparams
        
        self.hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        self.toolkit = TOOLKITS[control_method]()
        self.signs = None
        self.detector_coefs = None

        print('Hidden layers:', self.hidden_layers)
        print("\nController hyperparameters:")
        for n_, v_ in self.hyperparams.items():
            print(f"{n_:<20} : {v_}")
        print()

    def describe(self):
        def print_in_dashed_box(lines):
            # Determine the longest line for box width
            terminal_width = shutil.get_terminal_size().columns
            max_length = max(len(line) for line in lines)
            box_width = min(terminal_width, max_length + 4)

            # Print top border
            print('-' * box_width)

            # Print each line with padding and add dashed separator between lines
            for i, line in enumerate(lines):
                print(f"{line.ljust(box_width)}")
                if i < len(lines) - 1:  # Only add separator between lines, not after the last line
                    print('-' * box_width)

            # Print bottom border
            print('-' * box_width)

        lines = ['Controller Description:']
        for name, module in self.model.named_modules():
            lines.append(f"Model: {module}")
            break
        lines.append(f'Control method: {self.control_method}')
        lines.append(f'Tracked layers: {self.hidden_layers}')
            
        print_in_dashed_box(lines)

    def generate(self, prompt, layers_to_control=[], control_coef=0.4, **kwargs):
        if len(layers_to_control) == 0:
            control = False
        else:
            control = True     
            
        if control:               
            return self._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)
        else:
            return generation_utils.generate_on_text(self.model, self.tokenizer, prompt, **kwargs)
        
    def _controlled_generate(self, prompt, layers_to_control, control_coef, **kwargs):
        ## define hooks
        hooks = generation_utils.hook_model(self.model, self.directions, layers_to_control, control_coef)

        ## do forward pass
        out = generation_utils.generate_on_text(self.model, self.tokenizer, prompt, **kwargs)

        ## clear hooks
        generation_utils.clear_hooks(hooks)
        return out

    def compute_directions(self, train_data, train_labels, val_data=None, val_labels=None, hidden_layers=None, **kwargs):
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers
        self.directions, self.signs, self.detector_coefs, _ = self.toolkit._compute_directions(train_data, 
                                                                train_labels, 
                                                                val_data,
                                                                val_labels,
                                                                self.model, 
                                                                self.tokenizer, 
                                                                self.hidden_layers, 
                                                                self.hyperparams,
                                                                **kwargs
                                                            )
        
    def compute_directions_and_accs(self, 
                                    train_data, train_labels, 
                                    test_data, test_labels, 
                                    hidden_layers=None, **kwargs):
        
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers
        
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.tensor(test_labels).reshape(-1,1)
            
        
        self.directions, self.signs, self.detector_coefs, direction_accs = self.toolkit._compute_directions(
                                                           train_data, 
                                                           train_labels, 
                                                           self.model, 
                                                           self.tokenizer, 
                                                           self.hidden_layers, 
                                                           self.hyperparams,
                                                           test_data,
                                                           test_labels,
                                                           **kwargs
                                                          )
        
        return direction_accs
    
    def evaluate_directions(self,
                            train_data, train_labels,
                            val_data, val_labels,
                            test_data, test_labels,
                            hidden_layers=None, 
                            n_components=1,
                            agg_positions=False,
                            agg_model='linear',
                            layer_model='linear',
                            unsupervised=False,
                            selection_metric='auc',
                           ):
        
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.tensor(test_labels).reshape(-1,1)
        
        if len(train_labels.shape) == 1:
            train_labels = train_labels.unsqueeze(-1)
        if len(val_labels.shape) == 1:
            val_labels = val_labels.unsqueeze(-1)
        if len(test_labels.shape) == 1:
            test_labels = test_labels.unsqueeze(-1)
        
        train_y = train_labels.to(self.model.device).float()
        val_y = val_labels.to(self.model.device).float()
        test_y = test_labels.to(self.model.device).float()
        assert(train_y.shape[1]==val_y.shape[1]==test_y.shape[1])

        if not isinstance(train_data, dict):
            train_hidden_states = direction_utils.get_hidden_states(train_data, 
                                                                self.model, 
                                                                self.tokenizer, 
                                                                hidden_layers, 
                                                                self.hyperparams['forward_batch_size'],
                                                                all_positions=agg_positions
                                                                )
        else:
            train_hidden_states = train_data
            
        if not isinstance(val_data, dict):
            val_hidden_states = direction_utils.get_hidden_states(val_data, 
                                                                self.model, 
                                                                self.tokenizer, 
                                                                hidden_layers, 
                                                                self.hyperparams['forward_batch_size'],
                                                                all_positions=agg_positions
                                                                )
        else:
            val_hidden_states = val_data
        
        if not isinstance(test_data, dict):
            test_hidden_states = direction_utils.get_hidden_states(test_data, 
                                                              self.model, 
                                                              self.tokenizer, 
                                                              hidden_layers, 
                                                              self.hyperparams['forward_batch_size'],
                                                              all_positions=agg_positions
                                                             )
        else:
            test_hidden_states = test_data
        
        projections = {
                        'train' : [],
                        'val' : [],
                        'test' : []
                    }
        val_metrics = {}
        test_metrics = {}
        detector_coefs = {}
        test_predictions = {}
        for layer_to_eval in tqdm(hidden_layers):
            direction = self.directions[layer_to_eval]
            if isinstance(direction, np.ndarray):
                direction = torch.from_numpy(direction)
            direction = direction.to(self.model.device).float()[:n_components]
            direction = direction.T

            train_X = train_hidden_states[layer_to_eval].cuda().float()
            projected_train = train_X@direction

            val_X = val_hidden_states[layer_to_eval].cuda().float()
            projected_val = val_X@direction
            
            test_X = test_hidden_states[layer_to_eval].cuda().float()
            projected_test = test_X@direction
            
            if agg_positions:
                projected_val = torch.mean(projected_val, dim=1) # mean projection
                projected_test = torch.mean(projected_test, dim=1) # mean projection
    
            if layer_model == 'logistic':
                beta, b = direction_utils.logistic_solve(projected_val, val_y)
            elif layer_model == 'linear':
                beta, b = direction_utils.linear_solve(projected_val, val_y)
            else:
                raise ValueError(f"Invalid layer model: {layer_model}")
            
            detector_coefs[layer_to_eval] = [beta, b]
     
            if unsupervised: # evaluate sign on test data
                projected_test_preds = projected_test
                projected_test_preds = torch.where(projected_test_preds>0, 1, 0)
                
                projected_val_preds = projected_val
                projected_val_preds = torch.where(projected_val_preds>0, 1, 0)
            
            else: # evaluate slope, intercept on test data
                projected_val_preds = projected_val@beta + b
                projected_test_preds = projected_test@beta + b

            assert(projected_test_preds.shape==test_y.shape)
            test_predictions[layer_to_eval] = projected_test_preds
            
            val_metrics_on_layer = direction_utils.compute_prediction_metrics(projected_val_preds, val_y)
            val_metrics[layer_to_eval] = val_metrics_on_layer
            
            test_metrics_on_layer = direction_utils.compute_prediction_metrics(projected_test_preds, test_y)
            test_metrics[layer_to_eval] = test_metrics_on_layer
            
            projections['train'].append(projected_train.reshape(-1, n_components))
            projections['val'].append(projected_val.reshape(-1, n_components))
            projections['test'].append(projected_test.reshape(-1, n_components))
        
        agg_metrics, agg_beta, agg_bias, agg_predictions = direction_utils.aggregate_layers(projections, train_y, val_y, test_y, agg_model, 
                                                                                            tuning_metric=selection_metric)
        test_metrics['aggregation'] = agg_metrics
        test_predictions['aggregation'] = agg_predictions
        detector_coefs['aggregation'] = [agg_beta, agg_bias]

        best_layer_on_val = max(val_metrics, key=lambda x: val_metrics[x][selection_metric])
        test_predictions['best_layer'] = test_predictions[best_layer_on_val]
        test_metrics['best_layer'] = test_metrics[best_layer_on_val]
        
        return val_metrics, test_metrics, detector_coefs, test_predictions
    
    
    def detect(self, prompts, rep_layer=-15, use_rep_layer=False, use_avg_projection=False):
        hidden_states = direction_utils.get_hidden_states(
                            prompts, 
                            self.model, 
                            self.tokenizer, 
                            self.hidden_layers, 
                            self.hyperparams['forward_batch_size'],
                            all_positions=True
                         )
        
        projections = direction_utils.project_hidden_states(hidden_states, self.directions, self.hyperparams['n_components'])
        
        if use_avg_projection:
            scores = 0
            num_layers = 0
            for layer, h in projections.items():
                if layer!='agg' and layer>-21:
                    scores += h 
                    num_layers+=1
            
            preds = 0.5 + scores / num_layers # bias to mean 0.5
            
        elif 'agg' in self.detector_coefs and not use_rep_layer:
            preds = direction_utils.aggregate_projections_on_coefs(projections, self.detector_coefs['agg'])
            
        else:
            beta, b = self.detector_coefs[rep_layer]
            x = projections[rep_layer]
            preds = x@beta + b
            
        return preds.squeeze()
    
    def get_composite_directions(self,
                            val_data, val_labels,
                            n_components,
                            hidden_layers=None, 
                            agg_positions=False,
                            use_logistic=False
                           ):
        
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers
        
        val_y = torch.tensor(val_labels).to(self.model.device).float().reshape(-1,1)
        val_hidden_states = direction_utils.get_hidden_states(val_data, 
                                                              self.model, 
                                                              self.tokenizer, 
                                                              hidden_layers, 
                                                              self.hyperparams['forward_batch_size'],
                                                              all_positions=agg_positions
                                                             )
        
        composite_directions = {}
        
        for layer_to_eval in tqdm(hidden_layers):
            direction = self.directions[layer_to_eval]
            if isinstance(direction, np.ndarray):
                direction = torch.from_numpy(direction)
            direction = direction.to(self.model.device).float()[:n_components]
            direction = direction.T
            
            val_X = val_hidden_states[layer_to_eval].cuda().float()
            projected_val = val_X@direction
            
            beta = direction_utils.linear_solve(projected_val, val_y, use_bias=False)
            
            composite_vec = direction@beta
            composite_vec = composite_vec.reshape(1,-1)
            composite_directions[layer_to_eval] = composite_vec / composite_vec.norm()
                
        return composite_directions
        
    def save(self, concept, model_name, path='./', composite=False):
        if composite:
            filename = os.path.join(path, f'{self.control_method}_composite_{concept}_{model_name}.pkl')
        else:
            filename = os.path.join(path, f'{self.control_method}_{concept}_{model_name}.pkl')
            
        with open(filename, 'wb') as f:
            pickle.dump(self.directions, f)
            
            
        if self.detector_coefs is not None:
            detector_path = os.path.join(path, f'{self.control_method}_{concept}_{model_name}_detector.pkl')
            with open(detector_path, 'wb') as f:
                pickle.dump(self.detector_coefs, f)
            
    def load(self, concept, model_name, path='./', composite=False):
        if composite:
            filename = os.path.join(path, f'{self.control_method}_composite_{concept}_{model_name}.pkl')
        else:
            filename = os.path.join(path, f'{self.control_method}_{concept}_{model_name}.pkl')
        with open(filename, 'rb') as f:
            self.directions = pickle.load(f)
            self.hidden_layers = self.directions.keys()
        
        detector_path = os.path.join(path, f'{self.control_method}_{concept}_{model_name}_detector.pkl')
        if os.path.exists(detector_path):
            print("Detector found")
            with open(detector_path, 'rb') as f:
                self.detector_coefs = pickle.load(f)
        
    def format_prompt(self, prompt, steer=False):
        if self.name == 'toxicchat-t5-large':
            formatted_prompt = f"ToxicChat: {prompt}"
        else:
            chat = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=steer)
            formatted_prompt = formatted_prompt.strip('\n').strip()
        return formatted_prompt