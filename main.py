import random

# Simulate a large model with multiple "experts"
# In a real MoE model, these would be neural network modules
# with their own parameters.

def expert_color_analysis(data):
    """Expert specializing in color-based features."""
    print(f"  [Expert: Color Analysis] Processing data: '{data}'")
    # Simulate some computation specific to color analysis
    return f"Color processed for '{data}'"

def expert_shape_recognition(data):
    """Expert specializing in shape-based features."""
    print(f"  [Expert: Shape Recognition] Processing data: '{data}'")
    # Simulate some computation specific to shape recognition
    return f"Shape recognized for '{data}'"

def expert_texture_analysis(data):
    """Expert specializing in texture-based features."""
    print(f"  [Expert: Texture Analysis] Processing data: '{data}'")
    # Simulate some computation specific to texture analysis
    return f"Texture analyzed for '{data}'"

def expert_semantic_understanding(data):
    """Expert specializing in high-level semantic features."""
    print(f"  [Expert: Semantic Understanding] Processing data: '{data}'")
    # Simulate some computation for abstract understanding
    return f"Semantic understanding for '{data}'"

# A dictionary of all available experts. This represents the total
# 'parameter space' of our large simulated model.
ALL_EXPERTS = {
    "color": expert_color_analysis,
    "shape": expert_shape_recognition,
    "texture": expert_texture_analysis,
    "semantic": expert_semantic_understanding,
}

def router_gate(input_data):
    """
    Simulates a 'router' or 'gate' network that decides which experts to activate.
    In a real Mixture of Experts (MoE) model, this would be a small neural network
    that learns to output a probability distribution over experts, selecting the most relevant ones.
    """
    print(f"\n[Router Gate] Deciding which experts to activate for input: '{input_data}'")
    activated_experts = []

    # Simple routing logic based on input characteristics
    # This demonstrates that not all experts are needed for every input.
    if "apple" in input_data.lower():
        activated_experts.append("color") # Apples have distinct colors
        activated_experts.append("shape") # Apples have distinct shapes
        # 'texture' and 'semantic' experts are NOT activated for this input, saving computation.
    elif "orange" in input_data.lower():
        activated_experts.append("color") # Oranges have distinct colors
        activated_experts.append("texture") # Oranges have distinct textures (peel)
    elif "car" in input_data.lower():
        activated_experts.append("shape") # Cars have complex shapes
        activated_experts.append("semantic") # 'Car' requires higher-level understanding
    else:
        # For unknown inputs, activate a random subset or a default set
        print("  [Router Gate] Input not specifically recognized, activating a general subset.")
        activated_experts = random.sample(list(ALL_EXPERTS.keys()), k=random.randint(1, len(ALL_EXPERTS) - 1))

    print(f"  [Router Gate] Selected experts: {activated_experts}")
    return activated_experts

def process_input_with_moe(input_data):
    """
    Main function demonstrating sparse activation.
    It uses the router to select experts and then runs only those selected experts.
    """
    print(f"--- Processing Input: '{input_data}' ---")

    # The router decides which experts are relevant for this specific input.
    # This is the core of 'sparse activation': only a subset of the total model
    # parameters (represented by experts) are engaged for a given task.
    selected_expert_names = router_gate(input_data)

    results = {}
    # Only activate and run the selected experts. The unselected experts remain idle.
    for expert_name in selected_expert_names:
        if expert_name in ALL_EXPERTS:
            expert_function = ALL_EXPERTS[expert_name]
            results[expert_name] = expert_function(input_data)
        else:
            print(f"  [Warning] Expert '{expert_name}' not found.")

    print(f"\n--- Finished Processing '{input_data}' ---")
    print(f"Total experts available (total model capacity): {len(ALL_EXPERTS)}")
    print(f"Experts activated for this input (active parameters): {len(selected_expert_names)}")
    print(f"Results: {results}")
    print("-" * 40)
    return results

if __name__ == "__main__":
    print("Demonstrating Sparse Activation (Mixture of Experts) in a simulated AI model.")
    print("Even though the model has 4 'experts' (representing a large set of parameters),")
    print("only a subset is activated for each specific input, saving computation and resources.")
    print("=" * 60)

    # Test with different inputs to show varying expert activation
    process_input_with_moe("a red apple")
    process_input_with_moe("a juicy orange")
    process_input_with_moe("a fast car")
    process_input_with_moe("a green banana") # This will hit the 'else' in router_gate, demonstrating dynamic selection
    process_input_with_moe("a blue bird")
