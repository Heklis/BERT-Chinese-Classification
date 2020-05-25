import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

import logging
logging.basicConfig(level=logging.INFO)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    # Path to the TensorFlow checkpoint path.
    tf_checkpoint_path = "D:/Heklis/bin/chinese_L-12_H-768_A-12/bert_model.ckpt"
    # The config json file corresponding to the pre-trained BERT model. 
    # This specifies the model architecture.
    bert_config_file = "D:/Heklis/bin/chinese_L-12_H-768_A-12/bert_config.json"
    # Path to the output PyTorch model.
    pytorch_dump_path = "D:/Heklis/bin/chinese_L-12_H-768_A-12/pytorch.bin"
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path,
                                     bert_config_file,
                                     pytorch_dump_path)
