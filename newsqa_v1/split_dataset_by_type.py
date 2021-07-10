# coding: utf-8
import json

dataset_file = "combined-newsqa-data-v1.json"
train_file = "train-v1.0.json"
dev_file = "dev-v1.0.json"
test_file = "test-v1.0.json"


# Split newsqa dataset json file into ("train", "dev", "test") 3 type of json file according to "type" field.
def split_dataset_by_type(dataset_file, train_file, dev_file, test_file):
    raw_data = json.load(open(dataset_file, encoding="utf-8"))
    version, datalist = raw_data["version"], raw_data["data"]
    train_data, dev_data, test_data = [], [], []
    for d in datalist:
        if d["type"] == "train":
            train_data.append(d)
        elif d["type"] == "dev":
            dev_data.append(d)
        else:
            test_data.append(d)
    json.dump({"version": version, "data": train_data}, open(train_file, "w+", encoding="utf-8"))
    json.dump({"version": version, "data": dev_data}, open(dev_file, "w+", encoding="utf-8"))
    json.dump({"version": version, "data": test_data}, open(test_file, "w+", encoding="utf-8"))
    print("split dataset to {} train, {} dev and {} test.".format(len(train_data), len(dev_data), len(test_data)))


if __name__ == "__main__":
    split_dataset_by_type(dataset_file, train_file, dev_file, test_file)
