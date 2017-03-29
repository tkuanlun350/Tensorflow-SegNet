import tensorflow as tf
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "/tmp3/first350/TensorFlow/Logs", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "/tmp3/first350/SegNet-Tutorial/CamVid/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "/tmp3/first350/SegNet-Tutorial/CamVid/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "/tmp3/first350/SegNet-Tutorial/CamVid/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "20000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "360", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "480", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "11", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)

def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("CamVid testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)


def main(args):
    checkArgs()
    if FLAGS.testing:
        model.test(FLAGS)
    elif FLAGS.finetune:
        model.training(FLAGS, is_finetune=True)
    else:
        model.training(FLAGS, is_finetune=False)

if __name__ == '__main__':
  tf.app.run()
