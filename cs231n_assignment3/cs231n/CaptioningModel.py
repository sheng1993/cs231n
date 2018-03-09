import tensorflow as tf

class CaptioningModel(object):
    
    def __init__(self, word_to_idx:dict, dim_feature=512, dim_embed=512, dim_hidden=1024, n_time_steps=16):
        
        self.word_to_idx = word_to_idx
        self.V = len(word_to_idx)
        self.D = dim_feature
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_steps

        self._start = self.word_to_idx['<START>']
        self._end = self.word_to_idx['<END>']
        self._null = self.word_to_idx['<NULL>']

        # Default Initializers
        self.w_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Placeholder
        self.features = tf.placeholder(tf.float32, [None, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.image_idxs = tf.placeholder(tf.int32, [None])

    
    def _get_initial_lstm(self, features):
        '''
        Get initial LSTM state
        '''
        with tf.variable_scope('initial_lstm'):
            w_h = tf.get_variable('w_h', shape=[self.D, self.H], initializer=self.w_initializer)
            b_h = tf.get_variable('w_b', shape=[self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features, w_h) + b_h)

            w_c = tf.get_variable('w_c', shape=[self.D, self.H], initializer=self.w_initializer)
            b_c = tf.get_variable('b_c', shape=[self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features, w_c) + b_c)
            
            return h, c


    def _batch_norm(self, x, mode='train', name=None):
        '''
        Creates a batch normalization layer
        '''
        return tf.contrib.layers.batch_norm(inputs=x, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))


    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')
            return x

    
    def __decode_lstm(self, x, h, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.w_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.w_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            h_logits = tf.matmul(h, w_h) + b_h
            h_logits = tf.nn.tanh(h_logits)

            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits


    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(self.features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch normalization features
        features = self._batch_norm(features, mode='train', name='conv_features')

        # compute initial lstm states
        h, c = self._get_initial_lstm(features)

        # embedding lookup
        x = self._word_embedding(captions_in)

        loss = 0.0
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(x[:,t,:], state=[c, h]) # (c,h) , state=[c, h]
            logit = self.__decode_lstm(x[:,t,:], h, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=captions_out[:, t]) * mask[:, t])
        
        return loss / tf.to_float(batch_size)

    
    def build_sampler(self, max_len=20):
        features = self.features
        batch_size = batch_size = tf.shape(self.features)[0]

        # batch normalization features
        features = self._batch_norm(features, mode='test', name='conv_features')

        # compute initial lstm states
        h, c = self._get_initial_lstm(features)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        sampled_word = None
        sampled_word_list = []

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(x, state=[c, h])

            logits = self.__decode_lstm(x, h, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1,0))
        return sampled_captions