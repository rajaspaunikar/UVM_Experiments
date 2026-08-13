import torch

if torch.cuda.is_available():
    # Get total number of available GPUs
    device_count = torch.cuda.device_count()
    print(f"Total GPUs: {device_count}")
    
    # Fetch properties for the first GPU (device 0)
    props = torch.cuda.get_device_properties(0)
    
    print(f"Device Name: {props.name}")
    print(f"Total Memory: {props.total_memory / (1024**2):.2f} MB")
    print(f"CUDA Capability: {props.major}.{props.minor}")
    print(f"Multi-processor Count: {props.multi_processor_count}")
else:
    print("CUDA is not available.")