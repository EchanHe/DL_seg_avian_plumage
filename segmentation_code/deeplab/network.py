"""
## Purpose of script: The deeplab network
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers
slim = tf.contrib.slim
from resnet import resnet_v2, resnet_utils
import numpy as np
class Network:
    def __init__(self , config ,ori_img_width, ori_img_height , is_train = None):
        """
        Read in params to generate the neural network.
        """
        if is_train is not None:
            self.is_train = is_train
        else:
            self.is_train = config['is_train']
                
        self.is_grey = config['is_grey']
        self.network_name = config['network_name']

        self.cpu = '/cpu:0'     
        self.global_step = tf.get_variable("global_step", initializer=0,
                    dtype=tf.int32, trainable=False)
        self.batch_size = config['batch_size']

        if 'res_net_layers' in config:
            self.res_net_layers = config['res_net_layers']
        else:
            self.res_net_layers = "resnet_v2_50"

        if 'deeplab' in config:
        # deep lab version, v3 or v3plus
            self.deeplab_version = config['deeplab']
        else:
            self.deeplab_version = "v3"

        self.dropout_rate = config['dropout_rate']
        self.lambda_l2 = config["l2"]
        # self.stddev = config.stddev
        
        # self.use_fp16 = config.use_fp16
        self.class_num = 2
        self.params_dir = config['saver_directory']


        self.scale = config['scale']
        self.img_height = ori_img_height // self.scale
        self.img_width = ori_img_width // self.scale

        self.output_stride = config['output_stride']


        if self.is_train:
        # All learning rate decay is in `train_op(self, total_loss, global_step)`
            self.decay_restart = config["decay_restart"]
            if self.decay_restart is True:
                self.restart_decay_steps = config["first_decay_epoch"] * config["one_epoch_steps"]
                self.t_mul = _help_func_dict(config, 't_mul', 2.0)
                self.m_mul = _help_func_dict(config, 'm_mul', 1.0)            
            self.optimizer = _help_func_dict(config, 'optimizer', "adam")
            self.start_learning_rate =config["learning_rate"]
            self.exponential_decay_step = config["exponential_decay_epoch"] * config["one_epoch_steps"]
            self.learning_rate_decay = config["learning_rate_decay"]

        self.images = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width, config["input_channel"])
                )

        self.labels = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width))

        self.labels_by_classes = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width, self.class_num))

        # self.pred_images = tf.placeholder(
        #         dtype = tf.float32,
        #         shape = (None, self.img_height, self.img_width, 3)
        #         )

        print("Deeplab:")


        print("\nIs Training:{}\n\tInput shape: {}\n\tBatch size: {}\n\tOutput shape: {}".format(self.is_train,
         self.images.shape.as_list(),self.batch_size , self.labels_by_classes.shape.as_list()))
        
        if self.is_train:
            print("#### configuration ######")
            print("Optimizer: {}\tStart Learning rate: {}\tdecay_restart: {}".format(self.optimizer,
             self.start_learning_rate, self.decay_restart) )
        # self.miou = tf.placeholder(tf.float32, shape=(), name="valid_miou")
        # self.valid_miou = tf.summary.scalar("valid_miou", self.miou )
    def loss(self):
        """
        Return the loss function    
        """
        return tf.add( tf.add_n(tf.get_collection('losses'))  , 
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) , name = "total_loss")

    def add_to_loss(self , predicts ):
        """
        Goal: compute the loss with giving predicts

        params:
            predicts: The prediction tensor
    
        """

        valid_indices = tf.where(tf.not_equal(self.labels , -1))

        valid_labels = tf.gather_nd(params = self.labels_by_classes, 
            indices=valid_indices)

        valid_logits = tf.gather_nd(params=predicts, 
            indices=valid_indices)


        cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logits,
                                                                     labels=valid_labels)
        # #Weighted loss
        # # your class weights
        # class_weights = np.ones((1,self.class_num)) * 10.0
        # class_weights[0,0] = 1.0
        # # class_weights = tf.constant([[1.0, 1.0]])
        # # deduce weights for batch samples based on their true label
        # weights = tf.reduce_sum(class_weights * valid_labels, axis=1)
        # # print(valid_labels.shape)
        # weighted_losses = cross_entropies * weights
        # print("weight loss")

        cross_entropy_tf = tf.reduce_mean(cross_entropies)
        tf.add_to_collection("losses", cross_entropy_tf)

    def add_to_euclidean_loss(self, batch_size, predicts, labels, name):
        """
        将每一stage的最后一层加入损失函数 
        """

        flatten_vis = tf.reshape(self.vis_mask, [batch_size, -1])
        flatten_labels = tf.multiply( tf.reshape(labels, [batch_size, -1]) ,flatten_vis)
        flatten_predicts = tf.multiply(tf.reshape(predicts, [batch_size, -1]) , flatten_vis)
        # flatten_labels = tf.reshape(labels, [batch_size, -1])
        # flatten_predicts = tf.reshape(predicts, [batch_size, -1])
        # print(flatten_labels , flatten_predicts)
        with tf.name_scope(name) as scope:
            euclidean_loss = tf.sqrt(tf.reduce_sum(
              tf.square(tf.subtract(flatten_predicts, flatten_labels)), 1))
            # print(euclidean_loss)
            euclidean_loss_mean = tf.reduce_mean(euclidean_loss,
                name='euclidean_loss_mean')

        tf.add_to_collection("losses", euclidean_loss_mean)

    def train_op(self, total_loss, global_step):
        """
        Optimizer
        """
        self._loss_summary(total_loss)

        #####The learning rate decay method

        if self.decay_restart:
            # Cosine decay and restart
            # print("decayn restart: {}".format(self.restart_decay_steps))
            self.learning_rate = tf.train.cosine_decay_restarts(self.start_learning_rate, global_step,
             self.restart_decay_steps, t_mul = self.t_mul , m_mul = self.m_mul)
        else:
            # exponential_decay
            # print("expotineal decayn: {}".format(self.exponential_decay_step))
            self.learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                       self.exponential_decay_step, self.learning_rate_decay, staircase=True)

        ##### Select the optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        # grads = optimizer.compute_gradients(total_loss)
        # apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)  
          
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            apply_gradient_op = optimizer.minimize(total_loss, global_step)


        tf.summary.scalar("learning_rate", self.learning_rate, collections = ['train'])
        return apply_gradient_op


    ### Ckpt save and restore ####
    def save(self, sess, saver, filename, global_step):
        path = saver.save(sess, self.params_dir+filename, global_step=global_step)
        print ("Save params at " + path)

    def restore(self, sess, saver, filename):
        print ("Restore from previous model: ", self.params_dir+filename)
        saver.restore(sess, self.params_dir+filename)
    ## Save training log and summary ###
    def _loss_summary(self, loss):
        """
        Save the loss to summary log
        """
        with tf.device(self.cpu):
            with tf.name_scope('train_loss'):
                tf.summary.scalar(loss.op.name + "_raw", loss,collections=['train'])

    def _mIOU_summary(self, predicts):
        p = tf.reshape(tf.argmax(predicts, axis=3) , [self.batch_size,-1])
        l = tf.reshape(self.labels, [self.batch_size,-1])
        miou,self.update_op = tf.metrics.mean_iou(p, l, num_classes=self.class_num)
        # tf.summary.scalar('miou', miou)

    def _create_summary(self):
        """
        Goals: Create summaries for training and validation

        e.g, loss for training and valdation.
        Accuracy
        Images:
        """
        self.valid_loss = tf.placeholder(dtype = tf.float32)
        self.miou = tf.placeholder(dtype = tf.float32)
        self.precision = tf.placeholder(dtype = tf.float32)
        # self.point_pck = tf.placeholder(dtype = tf.float32,
        #  shape = (self.points_num,))        
        with tf.device(self.cpu):

            tf.summary.scalar("Valid_loss", self.valid_loss, collections = ['valid'])
            tf.summary.scalar("Valid_precision", self.precision, collections = ['valid'])
            tf.summary.scalar("Valid_IOU", self.miou, collections = ['valid'])

        self.valid_summary = tf.summary.merge_all('valid') 

    def _image_summary(self, x, channels):
        x = tf.cast(x, tf.float32)
        def sub(batch, idx):  
            name = x.op.name
            if channels>1:
                tmp = x[batch, :, :, idx] * 255
                tmp = tf.expand_dims(tmp, axis = 2)
                tmp = tf.expand_dims(tmp, axis = 0)
                tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100)
            else:

                tmp = x[batch, :, :] * 255
                tmp = tf.expand_dims(tmp, axis = 2)
                tmp = tf.expand_dims(tmp, axis = 0)
                tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100)
        for idx in range(channels):
            sub(0, idx)


    def _fm_summary(self, predicts):
      with tf.name_scope("fcn_summary") as scope:
          self._image_summary(self.labels, 1)
          tmp_predicts = tf.argmax(predicts , axis=3)
          self._image_summary(tmp_predicts, 1)


    @slim.add_arg_scope
    def atrous_spatial_pyramid_pooling(self,net, scope, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
        :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :param scope: scope name of the aspp layer
        :return: network layer with aspp applyed to it.
        """

        with tf.variable_scope(scope):
            feature_map_size = tf.shape(net)

            # apply global average pooling
            # image level feature
            image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
            image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                               activation_fn=None)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

            at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

            at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
            return net


    def deeplab_v3(self ):
        """
        # ImageNet mean statistics
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        """

        #~2000 expert project plumage images:
        # _R_MEAN = 45.48
        # _G_MEAN = 44.85
        # _B_MEAN = 44.29

        # 5094 genus bird photos
        _R_MEAN = 47.24
        _G_MEAN = 45.71
        _B_MEAN = 45.23

        uv_1_mean = 42.13
        uv_2_mean = 14.55
        uv_3_mean = 15.31

        # mean subtraction normalization
        inputs = self.images
        if inputs.shape[-1] ==3:
            inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]
        elif inputs.shape[-1] ==6:
            inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN , uv_1_mean, uv_2_mean, uv_3_mean]

        is_train = self.is_train

        # inputs has shape - Original: [batch, 513, 513, 3]
        with slim.arg_scope(resnet_utils.resnet_arg_scope(self.lambda_l2, is_train)):
            resnet = getattr(resnet_v2, self.res_net_layers)


            # _, end_points = resnet_v2.resnet_v2_50(inputs,
            #                        self.class_num,
            #                        is_training=is_train,
            #                        global_pool=False,
            #                        spatial_squeeze=False,
            #                        output_stride=self.output_stride)

            _, end_points = resnet(inputs,
                       self.class_num,
                       is_training=is_train,
                       global_pool=False,
                       spatial_squeeze=False,
                       output_stride=self.output_stride)

            with tf.variable_scope("DeepLab_v3"):

                # get block 4 feature outputs
                net = end_points[self.res_net_layers +'/block4']

                net = self.atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256)

                size = tf.shape(inputs)[1:3]
                # size = inputs.shape[1:3]
                if self.deeplab_version =='v3':
                    # version 3, bilinear upsample
                    net = slim.conv2d(net, self.class_num, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')


                    # # resize the output logits to match the labels dimensions
                    # #net = tf.image.resize_nearest_neighbor(net, size)
                    net = tf.image.resize_bilinear(net, size)
                    if is_train:
                        self.add_to_loss(net)
                        self._fm_summary(net)
                        self._create_summary()
                    return net

                elif self.deeplab_version == 'v3plus':
                    with tf.variable_scope("decoder"):
                        with tf.variable_scope("low_level_features"):
                            low_level_features = end_points[self.res_net_layers + '/block1/unit_3/bottleneck_v2/conv1']
                            low_level_features = slim.conv2d(low_level_features, 48,
                                                                   [1, 1], stride=1, scope='conv_1x1')
                            low_level_features_size = tf.shape(low_level_features)[1:3]

                        with tf.variable_scope("upsampling_logits"):
                            net = tf.image.resize_bilinear(net, low_level_features_size, name='upsample_1')
                            net = tf.concat([net, low_level_features], axis=3, name='concat')
                            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
                            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
                            net = slim.conv2d(net, self.class_num, [1, 1], activation_fn=None, 
                                normalizer_fn=None, scope='conv_1x1')
                            logits = tf.image.resize_bilinear(net, size, name='upsample_2')

                    if is_train:
                        self.add_to_loss(logits)
                        self._fm_summary(logits)
                        self._create_summary()

                    return logits            

# def training_network(prediction , loss , train_op, sess,config):
def _help_func_dict(config,key, default_value = None):
    if key in config:
        return config[key]
    else:
        return default_value