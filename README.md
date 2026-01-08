Using the python library num2words, I generated a dataset with 10,000 samples, 90% of which were used to train an AI. <br>
This specific AI will attempt to turn textual numbers ("two") to numerical numbers ("2").
<p></p>
The model only sees smaller numbers during training, and only larger numbers during testing, which is something that could be trained to improve the generalization. <br>
The number of epochs was also explicitly set to 10, which can be increased for better accuracy.
