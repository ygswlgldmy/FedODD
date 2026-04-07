"""FL: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedYogi

# from fl.task import Net
# from models import RTDETR_L as Net
from models import RTDETR_L_WithAttention as Net
pretrained_model_path = "/root/fl/models/YOLO/rtdetr-l.pt"
# pretrained_model_path = "/root/fl/final_model_pretrained.pt"
# pretrained_model_path = "/root/fl/Results/ultralytics/runs/detect/ODDNet-voc2007-125-2/weights/best.pt"
from fl.task import (
    load_rtdetr_weights,
    get_federated_model_record,
    load_federated_model_record,
    describe_fedbn_layout,
    DATASET_CONFIGS,
    CURRENT_DATASET,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Get the correct number of classes for the current dataset
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # Load global model with correct nc
    global_model = Net(nc=nc)
    global_model = load_rtdetr_weights(global_model, pretrained_model_path)

    freeze_layers = range(10)  # 对应 Backbone (HGStem 到 Stage 4)

    # for i in freeze_layers:
    #     # 访问 nn.Sequential 的特定层
    #     layer = global_model.model[i] 
    #     for param in layer.parameters():
    #         param.requires_grad = False

    # for param in global_model.parameters():
    #     param.requires_grad = False

    # for param in global_model.retnet_f5.parameters():
    #     param.requires_grad = True

    # for param in global_model.model[28].parameters():
    #     param.requires_grad = True

    arrays = get_federated_model_record(global_model)
    # arrays = ArrayRecord(global_model.state_dict())

    print(
        "FedBN server init: "
        f"{describe_fedbn_layout(global_model)}"
    )

    # Initialize FedAvg strategy
    
    # strategy = FedAvg(fraction_train=fraction_train)
    strategy = FedYogi(
        fraction_train=fraction_train,
        eta=1e-3,
        tau=1e-2,
        beta_1=0.8,
        beta_2=0.95,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    final_model = Net(nc=nc)
    final_model = load_rtdetr_weights(final_model, pretrained_model_path)
    load_federated_model_record(final_model, result.arrays)
    torch.save(final_model.state_dict(), "final_model.pt")
