from transformers import TFBertForSequenceClassification
import tensorflow as tf

#Maximal possible number of token sequence length
MAX_SEQ_LENGTH = 4096

class SBertModel(TFBertForSequenceClassification):
    """
        A helper class for using SBert models in transformers library
    """    

    def predict_from_text(self, text, tokenizer):
        """
            Runs the model and predicts the category of a text
        """            
        inputs = tokenizer(text, return_tensors='tf')
        prediction = self(**inputs)
        label_id = tf.argmax(prediction.logits, axis=-1).numpy()[0]
        score = max(prediction.logits.numpy()[0,:])

        return self.config.id2label[label_id], score

    def mean_pooling(self, model_output, attention_mask):        
        """
            Averages the token embeddings, taking input mask into account
        """                    
        token_embeddings = model_output[1][0]
        input_mask_expanded = tf.cast(
            tf.repeat(
                tf.expand_dims(attention_mask, -1),
                repeats=token_embeddings.shape[-1],
                axis=-1),
            tf.float32)
        return (
            tf.reduce_sum(token_embeddings * input_mask_expanded, axis=1) 
            / 
            tf.clip_by_value(tf.reduce_sum(input_mask_expanded, axis=1), clip_value_min=1e-9, clip_value_max=MAX_SEQ_LENGTH)
        )

    def encode(self, texts, tokenizer):
        """
            Generate embeddings (numeric vectors) of a text or list of texts
        """                            
        if type(texts) is not list:
            texts = [texts]
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
        model_output = self(**encoded_input, output_hidden_states=True)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])