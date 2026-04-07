"""FL: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# from fl.task import Net, load_data
# from models import RTDETR_L as Net
from models import RTDETR_L_WithAttention as Net
from fl.task import (
    get_federated_model_record,
    restore_client_local_state,
    cache_client_local_state,
    load_federated_model_record,
    describe_fedbn_layout,
    DATASET_CONFIGS,
    CURRENT_DATASET,
)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    load_federated_model_record(model, msg.content["arrays"])
    restore_client_local_state(model, partition_id)

    print(
        f"[Client {partition_id}] FedBN train load: "
        f"{describe_fedbn_layout(model)}"
    )

    # load_data 返回 trainloader, valloader, testloader, nc
    trainloader, _, _, nc_data = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        nc=nc,
    )

    # Construct and return reply Message
    cache_client_local_state(model, partition_id)
    model_record = get_federated_model_record(model)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    load_federated_model_record(model, msg.content["arrays"])
    restore_client_local_state(model, partition_id)

    print(
        f"[Client {partition_id}] FedBN eval load: "
        f"{describe_fedbn_layout(model)}"
    )

    # load_data 返回 trainloader, valloader, testloader, nc
    _, valloader, _, nc_data = load_data(partition_id, num_partitions)

    # Call the evaluation function — 用 val 集，传入 client_id 便于诊断
    eval_loss, eval_map50, eval_precision, eval_recall, eval_f1, _ = test_fn(
        model,
        valloader,
        device,
        nc=nc,
        client_id=partition_id,
        split_name="val",
    )

    # NaN 保护：确保上报的 metrics 都是有效 float
    def safe(v):
        return float(v) if (v == v and v >= 0) else 0.0

    # Construct and return reply Message
    metrics = {
        "eval_loss": safe(eval_loss),
        "eval_map50": safe(eval_map50),
        "eval_precision": safe(eval_precision),
        "eval_recall": safe(eval_recall),
        "eval_f1": safe(eval_f1),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
