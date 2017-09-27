import tensorflow as tf
import numpy as np
import time
import helper

class seq2seq():
    def __init__(self, vsize, esize, marker, slength, lrate=0.001, bsize=128, epoch=20, rsize=30, nlayer=2, kprob=1):
        self.vocab_source_size = vsize[0]
        self.vocab_target_size = vsize[1]
        self.embedding_enc_size = esize[0]
        self.embedding_dec_size = esize[1]
        self.startid = marker[0]
        self.endid = marker[1]
        self.max_source_sentence_length = slength
        self.learning_rate = lrate
        self.epochs = epoch
        self.batch_size = bsize
        self.rnn_size = rsize
        self.num_layers = nlayer
        self.keep_prob = kprob
       
        # Create TF Placeholders for input, target, learning_rate and keep_prob
        self.input_data = tf.placeholder(tf.int32, [None, None], name='input')
        self.target_data = tf.placeholder(tf.int32, [None, None], name='target')
        self.lrate = tf.placeholder(tf.float32, name='learning_rate')
        self.kprob = tf.placeholder(tf.float32, name='keep_prob')
        self.slength = tf.placeholder_with_default(self.max_source_sentence_length, None, name='sequence_length')
    
        #self.train_logits = None
        #self.inference_logits = None
        #self.cost = None
        #self.train_op = None
        #self.train_graph
        self.xxx = 1000
        #self.build_model()
   
    
    
        
    """ Preprocess target data for decoding """
    def process_decoding_input(self, target_data):
        temp = tf.strided_slice(target_data, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.startid), temp], 1)
        return dec_input
    
    
    """ Create a encode RNN layer """
    def encoding_layer(self, rnn_inputs, kprob):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=kprob)
        enc_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers)
        _, enc_state = tf.nn.dynamic_rnn(enc_cell, rnn_inputs, dtype=tf.float32)
        return enc_state
    
    
    """ Create a decode RNN layer """
    def decoding_layer(self, dec_embed_input, dec_embeddings, encoder_state, slength, kprob):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=kprob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers)
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, self.vocab_target_size, None, scope=decoding_scope)
        
        # Create training logits
        with tf.variable_scope("decoding") as decoding_scope:
            train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
            train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, train_decoder_fn, dec_embed_input, slength, scope=decoding_scope)
            train_logits = output_fn(train_pred)
            
        # Create inference logits
        with tf.variable_scope("decoding", reuse=True) as decoding_scope:
            infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, encoder_state, dec_embeddings, self.startid, self.endid, slength - 1, self.vocab_target_size)
            inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
        
        return train_logits, inference_logits
    
    
    
    def get_accuracy(self, target, logits):
        max_seq = max(target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(target, [(0,0),(0,max_seq - target.shape[1])], 'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(logits, [(0,0),(0,max_seq - logits.shape[1]), (0,0)], 'constant')
        return np.mean(np.equal(target, np.argmax(logits, 2)))
    
    
    
    """ Build seq2seq model """
    def seq2seq_model(self, input_data, target_data, slength, kprob):
   
        enc_embed_input = tf.contrib.layers.embed_sequence(input_data, self.vocab_source_size, self.embedding_enc_size)
        enc_state = self.encoding_layer(enc_embed_input, kprob)
    
        dec_input = self.process_decoding_input(target_data)
        dec_embeddings = tf.Variable(tf.random_uniform([self.vocab_target_size, self.embedding_dec_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        train_logits, inference_logits = self.decoding_layer(dec_embed_input, dec_embeddings, enc_state, slength, kprob)
        return train_logits, inference_logits
    


    #def train_op(self, train_source, train_target, valid_source, valid_target, save_path='checkpoints/dev'):
    def build_model(self):
        
        #self.train_graph = tf.Graph()
        #with self.train_graph.as_default():

            tf.reset_default_graph()
                         
            input_shape = tf.shape(self.input_data)
    
            self.train_logits, self.inference_logits = self.seq2seq_model(tf.reverse(self.input_data, [-1]), self.target_data, self.slength, self.kprob)
        
            #self.train_logits = train_logits
            #self.inference_logits = inference_logits

            tf.identity(self.inference_logits, 'logits')
            with tf.name_scope("optimization"):
                # Loss function
                self.cost = tf.contrib.seq2seq.sequence_loss(self.train_logits, self.target_data, tf.ones([input_shape[0], self.slength]))

                # Optimizer
                self.optimizer = tf.train.AdamOptimizer(self.lrate)

                # Gradient Clipping
                self.gradients = self.optimizer.compute_gradients(self.cost)
                self.capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(self.capped_gradients)
                
            print('xxxxxx')
            self.xxx = 10
            
        #print('yyyyyy')
        #print(self.xxx)
        #self.xxx = 100
 
    
 

    #def train_op(self, train_source, train_target, valid_source, valid_target, save_path='checkpoints/dev'):
    def train_op(self):   
    
        print('train:', self.xxx)
        
        
        
        """
        with tf.Session(graph=self.train_graph) as sess:
        #with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            print('hello')
            print(self.xxx)
        
            bsize = self.batch_size
            epoch = self.epochs

            for epoch_i in range(self.epochs):
                 for batch_i, (source_batch, target_batch) in enumerate(helper.batch_data(train_source, train_target, bsize)):
                     #start_time = time.time()
            
                     _, loss = sess.run([self.train_op, self.cost], {input_data: source_batch, target_data: target_batch, lrate: self.learning_rate, slength: target_batch.shape[1], kprob: self.keep_prob})
            
                     batch_train_logits = sess.run(self.inference_logits, {input_data: source_batch, kprob: 1.0})
                     batch_valid_logits = sess.run(self.inference_logits, {input_data: valid_source, kprob: 1.0})
                
                     train_acc = self.get_accuracy(target_batch, batch_train_logits)
                     valid_acc = self.get_accuracy(np.array(valid_target), batch_valid_logits)
                     #end_time = time.time()
                     if batch_i % 100 == 0:
                        print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                           .format(epoch_i, batch_i, len(train_source) // bsize, train_acc, valid_acc, loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print('Model Trained and Saved')
        """