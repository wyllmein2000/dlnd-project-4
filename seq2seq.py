from distutils.version import LooseVersion
import problem_unittests as tests
import warnings
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
   

    """ make sure you have the correct version of TensorFlow and access to a GPU """
    def check_gpu(self):
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        
        
    def unit_test(self):
        tests.test_sentence_to_seq(self.sentence_to_seq)
    
        
    def sentence_to_seq(self, sentence, vocab_to_int):
        """
        Convert a sentence to a sequence of ids
        :param sentence: String
        :param vocab_to_int: Dictionary to go from the words to an id
        :return: List of word ids
        """
        output = []
        for word in sentence.lower().split():
            if word in vocab_to_int:
                output.append(vocab_to_int[word])
            else:
                output.append(vocab_to_int['<UNK>'])
        return output    
        
        
        
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
    

    
 

    def train_op(self, train_source, train_target, valid_source, valid_target, save_path='checkpoints/dev'):
    
        #self.train_graph = tf.Graph()
        #with self.train_graph.as_default():
        tf.reset_default_graph()
        
        # Create TF Placeholders for input, target, learning_rate and keep_prob
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        target_data = tf.placeholder(tf.int32, [None, None], name='target')
        lrate = tf.placeholder(tf.float32, name='learning_rate')
        kprob = tf.placeholder(tf.float32, name='keep_prob')
        slength = tf.placeholder_with_default(self.max_source_sentence_length, None, name='sequence_length')
        
        
    
        train_logits, inference_logits = self.seq2seq_model(tf.reverse(input_data, [-1]), target_data, slength, kprob)
        
        tf.identity(inference_logits, 'logits')
        input_shape = tf.shape(input_data)

        
        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(train_logits, target_data, tf.ones([input_shape[0], slength]))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lrate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
                
                
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            start_time = time.time()
            
            bsize = self.batch_size
            epoch = self.epochs

            for epoch_i in range(self.epochs):
                 for batch_i, (source_batch, target_batch) in enumerate(helper.batch_data(train_source, train_target, bsize)):
                     
            
                     _, loss = sess.run([train_op, cost], {input_data: source_batch, target_data: target_batch, lrate: self.learning_rate, slength: target_batch.shape[1], kprob: self.keep_prob})
            
                     batch_train_logits = sess.run(inference_logits, {input_data: source_batch, kprob: 1.0})
                     batch_valid_logits = sess.run(inference_logits, {input_data: valid_source, kprob: 1.0})
                
                     train_acc = self.get_accuracy(target_batch, batch_train_logits)
                     valid_acc = self.get_accuracy(np.array(valid_target), batch_valid_logits)

                     if batch_i % 100 == 0:
                         print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'.format(epoch_i, batch_i, len(train_source) // bsize, train_acc, valid_acc, loss))

            end_time = time.time()

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print('Model Trained and Saved')
            print('Cost time: {} sec'.format(end_time - start_time))
        
        
        
        
    def translate(self, sentence, source_vocab_to_int, source_int_to_vocab, target_int_to_vocab, load_path):
       
        translate_sentence = self.sentence_to_seq(sentence, source_vocab_to_int)

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(load_path + '.meta')
            loader.restore(sess, load_path)

            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('logits:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

            translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

        print('Input')
        print('  Word Ids:      {}'.format([i for i in translate_sentence]))
        print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

        print('\nPrediction')
        print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
        print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
        output = [target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]
        return output