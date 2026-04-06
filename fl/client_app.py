"""FL: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# from fl.task import Net, load_data
# from models import RTDETR_L as Net
from models import RTDETR_L_WithAttention as Net
from fl.task import get_model_record, DATASET_CONFIGS, CURRENT_DATASET

from fl.task import load_data

from fl.task import test as test_fn
from fl.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Get the correct number of classes for the current dataset
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # Load the model and initialize it with the received weights
    model = Net(nc=nc)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # [修改] load_data 现在返回 trainloader, testloader, nc
    trainloader, _, nc_data = load_data(partition_id, num_partitions)

    # Call the training function
    # [修改] 传入 nc 参数
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        nc=nc,  # [新增] 传入类别数
    )

    # Construct and return reply Message
    model_record = get_model_record(model)

    # model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Get the correct number of classes for the current dataset
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # Load the model and initialize it with the received weights
    model = Net(nc=nc)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # [修改] load_data 现在返回 trainloader, testloader, nc
    _, valloader, nc_data = load_data(partition_id, num_partitions)

    # Call the evaluation function
    # [修改] 传入 nc 参数
    eval_loss, eval_map50, eval_precision, eval_recall, eval_f1, _ = test_fn(
        model,
        valloader,
        device,
        nc=nc,  # [新增] 传入类别数
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_map50": eval_map50,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
