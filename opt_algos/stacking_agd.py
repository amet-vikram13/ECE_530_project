import torch
import torch.nn as nn
import torch.optim as optim
import copy


def build_base_model(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)

def extend_model_with_layer(model, new_layer):
    if isinstance(model, nn.Sequential):
         new_model = nn.Sequential(*list(model.children()), new_layer)
    else:

         print("Warning: Model extension is conceptual. Adapt for your architecture.")

         if hasattr(model, 'blocks'): # e.g., a list of blocks
             model.blocks.append(new_layer)
             new_model = model # Modify in place
         else:
             raise NotImplementedError("Model structure not supported for conceptual extension.")
         return new_model
    return new_model


def initialize_layer_by_stacking(new_layer, source_layer):
    if new_layer.weight.shape == source_layer.weight.shape:
        with torch.no_grad():
            new_layer.weight.copy_(source_layer.weight)
            if hasattr(new_layer, 'bias') and new_layer.bias is not None:
                 new_layer.bias.copy_(source_layer.bias)
        print("Layer initialized by copying weights (stacking).")
    else:
        print("Warning: Layer shapes incompatible for stacking initialization.")


def train_model_stage(model, data_loader, optimizer, num_epochs_stage, device):
    model.train()

    for epoch in range(num_epochs_stage):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = torch.nn.functional.cross_entropy(outputs, targets) # Example loss
            loss.backward()
            optimizer.step()

    print(f"Finished training stage for {num_epochs_stage} epochs.")

def stacking_stagewise_training(
    input_dim,
    output_dim,
    num_stages,
    layers_per_stage, # How many layers to add per stage [15]
    num_epochs_per_stage,
    data_loader,
    learning_rate,
    device
):

    current_model = build_base_model(input_dim, output_dim).to(device)
    print("Stage 0: Initial base model created.")


    print(f"Stage 0: Training initial model for {num_epochs_per_stage} epochs.")

    optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
    train_model_stage(current_model, data_loader, optimizer, num_epochs_per_stage, device)
    print("Stage 0 training complete.")

    for stage in range(1, num_stages):
        print(f"\nStarting Stage {stage}/{num_stages - 1}...")


        source_layers = []

        if isinstance(current_model, nn.Sequential):
            source_layers = list(current_model.children())[-layers_per_stage:]

        elif hasattr(current_model, 'blocks'):
             source_layers = current_model.blocks[-layers_per_stage:]
        else:
            print("Warning: Cannot identify source layers for copying. Adapt 'stacking_stagewise_training'.")
            source_layers = [None] * layers_per_stage # Indicate random init


        new_layers = [build_base_model(input_dim, output_dim) for _ in range(layers_per_stage)]

        # Initialize new layers using stacking
        for i, new_layer in enumerate(new_layers):
             if source_layers[i] is not None: # Check if source was found
                initialize_layer_by_stacking(new_layer, source_layers[i])
             else:
                print(f"Layer {i} initialized randomly (no source found or incompatible).")
             new_layer.to(device) # Move to device after initialization


        for new_layer in new_layers:
             current_model = extend_model_with_layer(current_model, new_layer)

        print(f"Model extended with {layers_per_stage} new layers.")
        print(f"New total layers/blocks (conceptual): {len(list(current_model.children())) if isinstance(current_model, nn.Sequential) else 'Adapt count'}")



        optimizer = optim.Adam(current_model.parameters(), lr=learning_rate) # Example: Train all parameters


        train_model_stage(current_model, data_loader, optimizer, num_epochs_per_stage, device)
        print(f"Stage {stage} training complete.")

    print("\nStacking stagewise training finished.")
    return current_model