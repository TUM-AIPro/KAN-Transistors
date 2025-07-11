import torch
    
def DatasetQ(root_dir='data/', train_sample_step=1, name='qd', device='cpu'):

    assert train_sample_step in [1,2,4,10]

    gate_v_res = 165

    conditions = torch.load(root_dir+'input.pt')

    output = torch.load(root_dir+'output_{}.pt'.format(name)) # Already scaled by *1e+18

    dataset = {}

    if train_sample_step > 1:
        train_mask = torch.arange(0, gate_v_res, train_sample_step, dtype=torch.long)
        idx_mask = torch.arange(0, train_mask.size(0), 1, dtype=torch.long)
        train_mask_full = torch.empty(train_mask.size(0) ** 2, dtype=torch.long)
        for i in range(train_mask.size(0)):
            train_mask_full[idx_mask] = train_mask
            idx_mask += train_mask.size(0)
            train_mask += gate_v_res * train_sample_step
        
        dataset['train_input'] = conditions[train_mask_full,:].to(device)
        dataset['train_label'] = output[train_mask_full,0].view(-1,1).to(device)
        train_output = output[train_mask_full,:].to(device)
        dataset['train_deriv'] = train_output[:,[4,1,5,2,6,3]]
    
        test_mask_full = torch.ones(conditions.size(0), dtype=torch.bool)
        test_mask_full[train_mask_full] = 0
        dataset['test_input'] = conditions[test_mask_full,:].to(device)
        dataset['test_label'] = output[test_mask_full,0].view(-1,1).to(device)
        test_output = output[test_mask_full,:].to(device)
        dataset['test_deriv'] = test_output[:,[4,1,5,2,6,3]]
    else:
        dataset['train_input'] = dataset['test_input'] = conditions.to(device)
        dataset['train_label'] = dataset['test_label'] = output[:,0].view(-1,1).to(device)
        dataset['train_deriv'] = dataset['test_deriv'] = output[:,[4,1,5,2,6,3]].to(device)

    return dataset