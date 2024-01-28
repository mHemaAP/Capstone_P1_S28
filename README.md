# Capstone Project: Part 1 (S28 ERA V1)
## Train LLM from scratch ##

* We need to select a model that is less than 3B parameters (can be Microsoft's Phi 2 as well, but with random weights, hence training logs are MUST for capstone)
* Data:
  * It would be close to impossible to collect ALL the datasets required to train your model. Hence:
  * We are going to use Microsoft's Phi-2 or any other model and generate data. Recommendation was to generate this data in parallel, not generate and store everything as that would be a very very large dataset
  * We are going to collect "some" clean data (100MB when zipped). This data can be generated from Phi-2 and stored.
* Training
  * We are going to use the same tokenizer and other data structures to keep things simple
  * We are going to use AWS (or an equivalent system) where we are going to train the model. 
  * We are going to train the model (let's say that starts at 10). Train it somehow to reach the "initial loss - 1" value. Compare it with the final Microsoft's Phi 2's value and see how much more we have to train!!!

### Training Logs ###
```
Epoch: 0000 Step count:  -1 loss = 11.101167
		 Step count:  1 loss = 11.078608
		 Step count:  2 loss = 6.995222
		 Step count:  3 loss = 8.732363
		 Step count:  4 loss = 10.983467
		 Step count:  5 loss = 7.682126
		 Step count:  6 loss = 6.839418
		 Step count:  7 loss = 8.480166
		 Step count:  8 loss = 7.619623
		 Step count:  9 loss = 7.381527
		 Step count:  10 loss = 7.133785
Epoch: 0001 Step count:  10 loss = 8.751073
		 Step count:  11 loss = 7.391686
		 Step count:  12 loss = 9.393476
		 Step count:  13 loss = 9.144436
		 Step count:  14 loss = 8.941932
		 Step count:  15 loss = 8.992459
		 Step count:  16 loss = 3.553864
		 Step count:  17 loss = 8.735229
		 Step count:  18 loss = 8.871849
		 Step count:  19 loss = 7.911204
		 Step count:  20 loss = 8.087790
Epoch: 0002 Step count:  20 loss = 7.917281
		 Step count:  21 loss = 8.564671
		 Step count:  22 loss = 5.447293
		 Step count:  23 loss = 6.475473
		 Step count:  24 loss = 6.703707
		 Step count:  25 loss = 8.246754
		 Step count:  26 loss = 7.081114
		 Step count:  27 loss = 4.978745
		 Step count:  28 loss = 7.537660
		 Step count:  29 loss = 8.182168
		 Step count:  30 loss = 8.499785
Epoch: 0003 Step count:  30 loss = 6.702834
```

### Clean Dataset
Use clean dataset from here. Need to rename sample.mp4 to sample.zip
to use it.
[https://www.kaggle.com/datasets/medihemaap/transformer-clean-dataset-sample/](https://www.kaggle.com/datasets/medihemaap/transformer-clean-dataset-sample/data)

### data-feed-server.ipynb
This Jupyter notebook is responsible for generating data-feed for our model training.
The following piece of code either generates text from a pre-trained transformer (Phi-2) model or reads randomly from sample.zip file
every second and keeps on adding it to a Queue asynchronously.

```
loop = asyncio.get_event_loop()
import queue
q = queue.Queue()

def callback(flag=False):
    print("Adding Text To Queue", flag)
    if flag:
        idx = random.randint(1, tokenizer_length)
        token = tokenizer.decode([idx])
        q.put( generate_text(token) )
        #q.put( read_text() )
    else:
        q.put( read_text() )
    
    
    if q.qsize() > 100:
        flag = True
    else:
        flag = False
    
    loop.call_later(0.01, callback, flag)
    #callback()
    
            
callback()
```

And the following code snippet starts an ngrok-web-server, and exposes server \<Public URL\>

```
port = 8000
ngrok_tunnel = ngrok.connect(port, pyngrok_config=conf.PyngrokConfig(auth_token=Ngrok_token))

# where we can visit our fastAPI app
print('Public URL:', ngrok_tunnel.public_url)


nest_asyncio.apply()

# finally run the app
uvicorn.run(app, port=port)
```

**Note:** <ins>_The \<Public URL\> changes every time the code is run_</ins>

```
INFO:     Started server process [127]
INFO:     Waiting for application startup.
Public URL: https://e60f-35-233-217-40.ngrok-free.app
```

when \<Public URL\>/generate API is called it returns an array of texts.
These texts are either from the stored text-files (sample.zip) or generated from a pre-trained Phi-2 model.

## training.ipynb
This training notebook is run on colab to train the model. please note that \<Public URL\>/generate needs to be added as **feed_url** in
_config.py_ file 

## phi_train_clean_data
This folder contains the notebook trained for Phi1.5 and Phi2 models for a clean data sample taken from redpajama data. The corresponding logs as follows -

### Training Microsoft Phi2 GPT model from scratch

#### Features:
1. Phi2 Model - https://huggingface.co/microsoft/phi-2
2. Trained on 4 A100 80 GB GPU Ram.
3. Loss reduced from 11 to 4.
4. Used 100 MB zipped clean data - wikipedia, archive and book. Sample from https://github.com/togethercomputer/RedPajama-Data

### Training Logs ###
```
{'model_name': 'phi-2', 'name': 'phi2_gpt', 'save_interval': 1000, 'eval_interval': 1000, 'eval_iters': 100, 'log_interval': 10, 'learning_rate': 0.006, 'batch_size': 8, 'micro_batch_size': 8, 'gradient_accumulation_steps': 1, 'max_iters': 600000, 'weight_decay': 0.1, 'beta1': 0.9, 'beta2': 0.95, 'grad_clip': 1.0, 'decay_lr': True, 'warmup_iters': 2000, 'lr_decay_iters': 600000, 'min_lr': 6e-06}

Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
[rank: 2] Seed set to 1337
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
[rank: 3] Seed set to 1337
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
[rank: 1] Seed set to 1337

---------------------------------------------------------------------------------------------------- 
distributed_backend=nccl All distributed processes registered. Starting with 4 processes

----------------------------------------------------------------------------------------------------
[rank: 0] Seed set to 1337
Loading model with {'name': 'phi-2', 'hf_config': {'org': 'microsoft', 'name': 'phi-2'}, 'block_size': 2048, 'vocab_size': 50257, 'padding_multiple': 512, 'padded_vocab_size': 51200, 'n_layer': 32, 'n_head': 32, 'n_embd': 2560, 'rotary_percentage': 0.4, 'parallel_residual': True, 'bias': True, 'lm_head_bias': True, 'n_query_groups': 32, 'shared_attention_norm': True, '_norm_class': 'LayerNorm', 'norm_eps': 1e-05, '_mlp_class': 'GptNeoxMLP', 'gelu_approximate': 'tanh', 'intermediate_size': 10240, 'rope_condense_ratio': 1, 'rope_base': 10000, 'head_size': 80, 'rope_n_elem': 32} 
Time to instantiate model: 0.16 seconds. 
Total parameters 2,779,683,840 
Estimated TFLOPs: 1312.92 
Measured TFLOPs: 1173.04 
iter 0 step 1: loss 11.1288, LR: 0.000000, iter time: 2820.98ms (optimizer.step) 
iter 10 step 11: loss 6.6115, LR: 0.000030, iter time: 2432.98ms (optimizer.step) 
iter 20 step 21: loss 4.9748, LR: 0.000060, iter time: 2433.09ms (optimizer.step) 
iter 30 step 31: loss 3.5289, LR: 0.000090, iter time: 2431.82ms (optimizer.step) 
iter 40 step 41: loss 7.5937, LR: 0.000120, iter time: 2434.67ms (optimizer.step) 
iter 50 step 51: loss 5.7037, LR: 0.000150, iter time: 2435.77ms (optimizer.step) 
iter 60 step 61: loss 5.0583, LR: 0.000180, iter time: 2481.01ms (optimizer.step) 
iter 70 step 71: loss 4.5447, LR: 0.000210, iter time: 2435.36ms (optimizer.step) 
iter 80 step 81: loss 3.1150, LR: 0.000240, iter time: 2434.76ms (optimizer.step) 
iter 90 step 91: loss 5.3811, LR: 0.000270, iter time: 2435.89ms (optimizer.step) 
iter 100 step 101: loss 4.3036, LR: 0.000300, iter time: 2437.67ms (optimizer.step) 
iter 110 step 111: loss 4.2830, LR: 0.000330, iter time: 2438.59ms (optimizer.step) 
iter 120 step 121: loss 4.6569, LR: 0.000360, iter time: 2439.37ms (optimizer.step) 
iter 130 step 131: loss 4.6091, LR: 0.000390, iter time: 2497.76ms (optimizer.step) 
iter 140 step 141: loss 5.1293, LR: 0.000420, iter time: 2439.01ms (optimizer.step) 
iter 150 step 151: loss 4.6357, LR: 0.000450, iter time: 2443.62ms (optimizer.step) 
iter 160 step 161: loss 5.0636, LR: 0.000480, iter time: 2446.68ms (optimizer.step) 
iter 170 step 171: loss 4.3482, LR: 0.000510, iter time: 2440.68ms (optimizer.step) 
iter 180 step 181: loss 4.6043, LR: 0.000540, iter time: 2441.63ms (optimizer.step) 
iter 190 step 191: loss 3.1436, LR: 0.000570, iter time: 2440.67ms (optimizer.step) 
iter 200 step 201: loss 4.0853, LR: 0.000600, iter time: 2443.81ms (optimizer.step)
```
## Comparing the Training Logs - Clean data sample Vs Stream data sample ###
The average loss while training Phi2 model for clean data sample alone at iteration 30 and step 31 is found to be 3.5289 whereas the average loss while training Phi2 model for stream data sample at the same iteration/step is found to be 6.702834. This makes it clear that training the phi2 model with clean data gives better performance. This is that proof that if we were to have more resources we will be able to train it further for better loss / performance.
