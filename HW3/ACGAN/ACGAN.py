
import os
import time
from ops import *
from utils import *
import tensorflow as tf
import numpy as np
from inception_score import *
from matplotlib import pyplot
import matplotlib.pyplot as plt































class ACGAN(object):
    model_name = "ACGAN"

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size


        if dataset_name == 'cifar-10':
            self.data_X, self.data_y = load_cifar(self.dataset_name)

        else:
            raise NotImplementedError


        self.input_height = 32
        self.input_width = 32
        self.output_height = 32
        self.output_width = 32

        self.z_dim = z_dim
        self.y_dim = 10
        self.c_dim = 3


        self.learning_rate_D = 0.0002
        self.learning_rate_G = 0.001
        self.learning_rate_Q = 0.0001
        self.beta1 = 0.5


        self.sample_num = 64

        self.len_discrete_code = 10


        self.num_batches = len(self.data_X) // self.batch_size

    def classifier(self, x, is_training=True, reuse=False):

        with tf.variable_scope("classifier", reuse=reuse):

            net = lrelu(bn(linear(x, 128, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            out_logit = linear(net, self.y_dim, scope='c_fc2')
            out = tf.nn.softmax(out_logit)

            return out, out_logit

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            net = lrelu(conv2d(net, 1024, 4, 4, 4, 4, name='d_conv4'))
            net = tf.reshape(net, [self.batch_size, -1])
            out_logit = linear(net, 1, scope='d_fc5')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, y, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):

            z = concat([z, y], 1)

            net = tf.nn.relu(bn(linear(z, 2*2*448, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.reshape(net, [self.batch_size, 2, 2, 448])
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 4, 4, 256], 4, 4, 2, 2, name='g_dc2'),
                                is_training=is_training,scope='g_bn2'))
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 128], 4, 4, 2, 2, name='g_dc3'),
                                is_training=is_training, scope='g_bn3'))
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 16, 16, 64], 4, 4, 2, 2, name='g_dc4'),
                                is_training=is_training, scope='g_bn4'))
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 32, 32, 3], 4, 4, 2, 2, name='g_dc5'))

            return out

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """

        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')


        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')


        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        D_real, D_real_logits, input4classifier_real = self.discriminator(self.inputs, is_training=True, reuse=False)


        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, input4classifier_fake = self.discriminator(G, is_training=True, reuse=True)


        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake


        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))


        code_fake, code_logit_fake = self.classifier(input4classifier_fake, is_training=True, reuse=False)
        code_real, code_logit_real = self.classifier(input4classifier_real, is_training=True, reuse=True)


        q_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_logit_real, labels=self.y))


        q_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_logit_fake, labels=self.y))


        self.q_loss = q_fake_loss + q_real_loss


        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate_D, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate_G, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)
            self.q_optim = tf.train.AdamOptimizer(self.learning_rate_Q, beta1=self.beta1) \
                .minimize(self.q_loss, var_list=q_vars)


        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)


        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        q_loss_sum = tf.summary.scalar("g_loss", self.q_loss)
        q_real_sum = tf.summary.scalar("q_real_loss", q_real_loss)
        q_fake_sum = tf.summary.scalar("q_fake_loss", q_fake_loss)


        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.q_sum = tf.summary.merge([q_loss_sum, q_real_sum, q_fake_sum])

    def train(self):

        tf.global_variables_initializer().run()


        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_codes = self.data_y[0:self.batch_size]


        self.saver = tf.train.Saver()


        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        IS = []


        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):


            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)


                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.y: batch_codes,
                                                                  self.z: batch_z})
                self.writer.add_summary(summary_str, counter)


                _, summary_str_g, g_loss, _, summary_str_q, q_loss = self.sess.run(
                    [self.g_optim, self.g_sum, self.g_loss, self.q_optim, self.q_sum, self.q_loss],
                    feed_dict={self.z: batch_z, self.y: batch_codes, self.inputs: batch_images})
                self.writer.add_summary(summary_str_g, counter)
                self.writer.add_summary(summary_str_q, counter)


                counter += 1
                if counter%500==0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, q_loss))


                if np.mod(counter, 500) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], './' + check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                        epoch, idx))

            start_batch_id = 0

            if epoch%5 == 0:

                self.save(self.checkpoint_dir, counter)

                self.visualize_results(epoch)

                [a, b] = self.calculate_is()
                print('\n',a, b,'\n')
                IS.append(a)

        self.save(self.checkpoint_dir, counter)

        N = len(IS)
        x = np.linspace(0, 5 * N - 5, N)
        plt.plot(x, IS)
        plt.xlabel('epoch')
        plt.ylabel("IS")
        plt.savefig(check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + 'IS.png',dpi = 900)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        """ random noise, random discrete code, fixed continuous code """
        y = np.random.choice(self.len_discrete_code, self.batch_size)
        y_one_hot = np.zeros((self.batch_size, self.y_dim))
        y_one_hot[np.arange(self.batch_size), y] = 1

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})

        save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10

        np.random.seed()
        si = np.random.choice(self.batch_size, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.batch_size, dtype=np.int64) + l
            y_one_hot = np.zeros((self.batch_size, self.y_dim))
            y_one_hot[np.arange(self.batch_size), y] = 1

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.len_discrete_code],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    def calculate_is(self):
        imgs = np.zeros((((self.batch_size*100,32,32,3))))
        for k in range(10):
            for i in range(10):
                y = np.zeros((self.batch_size, 10))
                y[:, i] = 1
                z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                image = self.sess.run(self.fake_images, feed_dict = {self.z:z, self.y:y})
                imgs[self.batch_size*10*k+self.batch_size*i:self.batch_size*10*k+self.batch_size*i+self.batch_size,:,:,:] = image
        imgs = np.transpose(imgs, axes=[0, 3, 1, 2])
        imgs = (imgs-0.5)*2
        return (inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
