from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="data.yaml path"
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--name", type=str, default="yolov8n", help="renames results.txt"
    )
    parser.add_argument(
        "--yolo", type=str, default="yolov8n.pt", help="yolo model type"
    )
    args = parser.parse_args()

    config = args.config
    num_epoch = args.epochs
    device = args.device
    name = args.name
    yolo = args.yolo

    # load the model
    model = YOLO(yolo)

    # Train the model
    model.train(data=config, epochs=num_epoch, device=device, patience=0, name=name)

    # validate the model
    # model = YOLO('runs\\detect\\temp\\weights\\best.pt')
    # metric = model.val(name='temp_val')
