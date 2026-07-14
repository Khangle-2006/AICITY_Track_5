import torch

def main():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_properties(i).name
        print(f"GPU {i}: {name} | Free: {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")

if __name__ == "__main__":
    main()
