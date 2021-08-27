#  Self-paced ensemble learning for speech and audio classification                                                                                    

We propose a self-paced ensemble learning scheme in which models learn from each other over several iterations. 
During the self-paced learning process based on pseudo-labeling, in addition to improving the individual models, 
our ensemble also gains knowledge about the target domain. 
Our empirical results indicate that SPEL significantly outperforms the baseline ensemble models. 

-----------------------------------------                                                                                                                                      
### About
In this project you can reproduce the results for the Rainforest Connection Species Audio Detection 
(https://www.kaggle.com/c/rfcx-species-audio-detection) data set. We added the SPEL algorithm for two
different networks: ResNet and ResNeSt. 

```
> In order to work properly you need to have a python version older than 3.6
>> We used the following version 3.6.8
```

## Run experiments

In order to run the SPEL experiment you should do:

 - Run from main.py "train_ensamble" method (on config.json you should have "add_spl_data" -> false)
 - Run from main.py "generate_spl_data" method
 - Run again from main.py "train_ensamble" method (on config.json you should have "add_spl_data" -> true)
 - Run from main.py "generate_spl_data" method
 and so on...


Warning: The SPEL data is generated in "spel_data_path" directory (which is set in config.json). 
When you run again the training, all data from directory will be considered at training. 
Is up to you if you want to remove from them or let them all.


### SPEL generation
The SPEL data are generated in accordance with the predicted score. 
If the score is over "spl_th" (from config.json) the example is considered in SPEL data set.


## Cite us

```
@article{ristea2021self,
  title={Self-paced ensemble learning for speech and audio classification},
  author={Ristea, Nicolae-Catalin and Ionescu, Radu Tudor},
  booktitle={Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH},
  volume={2021},
  year={2021}
}
```

## You can send your questions or suggestions to: 
r.catalin196@yahoo.ro, raducu.ionescu@gmail.com

### Last Update:
August 27, 2021 



