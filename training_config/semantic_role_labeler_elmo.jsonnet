{
    "dataset_reader": {
        "type": "srl",
        "token_indexers": {
            "elmo_chars": {
                "type": "elmo_characters",
		"char_map":"/homes/gws/pmulc/multilingual_lm/models/elmo_wordchartest/char_vocab.txt"
            }
        }
    },
    "train_data_path": "/homes/gws/pmulc/data/ontonotes/conll-2012/v4/data/train/data/chinese/annotations/",
    "validation_data_path": "/homes/gws/pmulc/data/ontonotes/conll-2012/v4/data/development/data/chinese/annotations/",
    "model": {
        "type": "srl",
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "/data/pmulc/polyglot_elmo/models/elmo_cmns/options_new.json",
                    "weight_file": "/data/pmulc/polyglot_elmo/models/elmo_cmns/weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.1
                }
            },
	    "allow_unmatched_keys": true,
	    "embedder_to_indexer_map": {
	        "elmo": ["elmo_chars", "elmo_tokens"]
	        "tokens": ["tokens"]
	        "token_characters": ["token_characters"]
	    }
        },
        "initializer": [
            [
                "tag_projection_layer.*weight",
                {
                    "type": "orthogonal"
                }
            ]
        ],
        // NOTE: This configuration is correct, but slow.
        // If you are interested in training the SRL model
        // from scratch, you should use the 'alternating_lstm_cuda'
        // encoder instead.
        "encoder": {
            "type": "alternating_lstm",
            "input_size": 1124,
            "hidden_size": 300,
            "num_layers": 8,
            "recurrent_dropout_probability": 0.1,
            "use_input_projection_bias": false
        },
        "binary_feature_dim": 100,
        "regularizer": [
            [
                ".*scalar_parameters.*",
                {
                    "type": "l2",
                    "alpha": 0.001
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ],
        "batch_size": 80
    },
    "trainer": {
        "num_epochs": 500,
        "grad_clipping": 1.0,
        "patience": 200,
        "num_serialized_models_to_keep": 10,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        }
    }
}
