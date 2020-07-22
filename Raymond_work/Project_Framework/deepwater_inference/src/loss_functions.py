import tensorflow as tf


def cross_entropy_balanced(y_true, y_pred):
    ''' Cross Entropy weighted by a weight mask
    shape of y_true (m, n, 1)
    shape of y_pred (m, n, 2)
    y_true  - GT of weghted loss
            - 1st layer / segmentation mask, {0,1}
            - 2nd layer / pixel weights (1, 10)
    y_pred  - computed segmentation probabilities
            - 1st layer / background (0,1)
            - 2nd layer / foreground (0,1)
    '''

    epsilon: float = 0.000001

    foreground, pixel_weights = tf.split(y_true, 2, 3)
    prob_background, prob_foreground = tf.split(y_pred, 2, 3)

    y_false = (1 - y_true)

    sum_foreground: float = tf.reduce_sum(y_true * pixel_weights) + epsilon
    sum_background: float = tf.reduce_sum(y_false * pixel_weights) + epsilon

    class_balance_mask = pixel_weights * (prob_foreground / (2 * sum_foreground) +
                                          prob_background / (2 * sum_background))

    loss = class_balance_mask * (y_false * tf.math.log(prob_background + epsilon) +
                                 y_true * tf.math.log(prob_foreground + epsilon))

    return - tf.reduce_sum(loss)


def w_cren_2ch_bala_m(y_true, y_pred):
    ''' Cross Entropy weighted by a weight mask
    shape of y_true (m, n, 3)
    shape of y_pred (m, n, 4)
    y_true  - GT of weghted loss
            - 1st layer / segmentation mask, {0,1}
            - 2nd layer / weights <1, 4>
            - 3rd layer / borders {0,1}
    y_pred  - computed segmentation
            - 1st layer / segmentation (0,1)
            - 2nd layer / borders (0,1)
    '''

    LAMBDA = 0.0001

    m_true, w_true = tf.split(y_true, 2, 3)
    m0_prob, m1_prob = tf.split(y_pred, 2, 3)

    m_false = (1 - m_true)

    sum_Fm = tf.reduce_sum(m_true * w_true) + LAMBDA
    sum_Bm = tf.reduce_sum(m_false * w_true) + LAMBDA

    w_m_bala = w_true * m_true / (2 * sum_Fm) + w_true * m_false / (2 * sum_Bm)

    loss_m = w_m_bala * (m_false * tf.math.log(m0_prob + LAMBDA) + m_true * tf.math.log(m1_prob + LAMBDA))

    loss = - tf.reduce_mean(tf.reduce_sum(loss_m, axis=[1, 2, 3]), axis=0)

    return loss


def w_cren_2ch_bala(y_true, y_pred):
    ''' Cross Entropy weighted by a weight mask
    shape of y_true (m, n, 3)
    shape of y_pred (m, n, 4)
    y_true  - GT of weghted loss
            - 1st layer / segmentation mask, {0,1}
            - 2nd layer / weights <1, 4>
            - 3rd layer / borders {0,1}
    y_pred  - computed segmentation
            - 1st layer / segmentation (0,1)
            - 2nd layer / borders (0,1)
    '''

    LAMBDA = 0.0000001

    m_true, _ = tf.split(y_true, 2, 3)
    m0_prob, m1_prob = tf.split(y_pred, 2, 3)

    m_false = (1 - m_true)

    sum_Fm = tf.reduce_sum(m_true) + LAMBDA
    sum_Bm = tf.reduce_sum(m_false) + LAMBDA

    w_m_bala = m_true / (2 * sum_Fm) + m_false / (2 * sum_Bm)

    loss_m = w_m_bala * (m_false * tf.math.log(m0_prob + LAMBDA) + m_true * tf.math.log(m1_prob + LAMBDA))

    # loss = - tf.reduce_mean(tf.reduce_sum(loss_m, axis=[1, 2, 3]), axis=0)
    loss = - tf.reduce_sum(loss_m)

    return loss
