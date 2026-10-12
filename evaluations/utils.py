import ast

def extract_shots_from_mapping(mapping_str):
    """
    Extract selected and all shots from a mapping string.
    
    Args:
        mapping_str: String representation of a dictionary mapping
        
    Returns:
        tuple: (selected_shots, all_shots) as lists of integers
    """
    mapping_dict = ast.literal_eval(mapping_str)
    
    selected_shots = [int(k.split('_')[1]) for k, v in mapping_dict.items() if v is not None]
    all_shots = [int(k.split('_')[1]) for k in mapping_dict.keys()]
    
    return selected_shots, all_shots


def create_one_hot(all_shots, selected_shots):
    """
    Create a one-hot vector indicating which shots are selected.
    
    Args:
        all_shots: List of all shot indices
        selected_shots: List of selected shot indices
        
    Returns:
        list: One-hot vector where 1 indicates the shot is selected
    """
    selected_set = set(selected_shots)  # O(1) lookup
    one_hot = [1 if shot in selected_set else 0 for shot in all_shots]
    
    return one_hot

