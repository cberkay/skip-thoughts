"""
Model specification
"""
import theano
import theano.tensor as tensor

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import norm_weight
from layers import get_layer

def init_params(options, preemb=None, preinit=None, predec=None, preouthid=None,
                preoutlogit=None):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # HACK:
    options['n_clusters'] = 10
    options['dim_char'] = 4096
    #options['dim_ctx'] = options['dimctx']

    # Cluster embedding
    params['Cemb'] = norm_weight(options['n_clusters'], options['dim_char'])

    # Word embedding
    if preemb == None:
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    else:
        params['Wemb'] = preemb

    # HACK
    if not 'dim_ctx' in options:
        options['dim_ctx'] = options['dimctx']

    # Pre-Initial state
    params = get_layer('ff')[0](options, params, prefix='pre_ff_state',
                                nin=options['dim_char'], nout=options['dim_ctx'],
                                weights=preinit)

    # Initial state
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=options['dim_ctx'], nout=options['dim'],
                                weights=preinit)

    # Decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              weights=predec)

    # Output layer
    if options['doutput']:
        params = get_layer('ff')[0](options, params, prefix='ff_hid',
                                    nin=options['dim'],
                                    nout=options['dim_word'],
                                    weights=preouthid)
        params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                    nin=options['dim_word'],
                                    nout=options['n_words'],
                                    weights=preoutlogit)
    else:
        if preouthid:
            print "The parameters provided for the output hidden are unused."
        params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                    nin=options['dim'], nout=options['n_words'],
                                    weights=preoutlogit)

    return params

def build_model(tparams, options):
    """
    Computation graph for the model
    """
    trng = RandomStreams(1234)

    # description string: #words x #samples; identifies the words
    x = tensor.matrix('x', dtype='int64')

    # description string mask: #words x #samples;
    # which words are to be used, as dictated by the length of the sentence
    mask = tensor.matrix('mask', dtype='float32')

    # the encoded source (sentence, image, etc.): #samples x dim_ctx
    ctx = tensor.matrix('ctx', dtype='float32')

    # cluster indices: 1 x #samples
    c_idc = tensor.vector('c_idc', dtype='int64')

    # flatten to 1 x (#words * #samples)
    x_flat = x.flatten()

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Index into the cluster embedding matrix
    Cemb = tparams['Cemb'][c_idc].reshape([n_samples, options['dim_char']])

    # Index into the word embedding matrix, shift it forward in time
    Wemb = tparams['Wemb'][x_flat].reshape([n_timesteps, n_samples, options['dim_word']])
    Wemb_shifted = tensor.zeros_like(Wemb)
    Wemb_shifted = tensor.set_subtensor(Wemb_shifted[1:], Wemb[:-1])
    Wemb = Wemb_shifted

    # Preinit state
    preinit_state = get_layer('ff')[1](tparams, Cemb, options, prefix='pre_ff_state', activ='tanh')

    # Init state
    init_state = get_layer('ff')[1](tparams, ctx + preinit_state, options, prefix='ff_state', activ='tanh')

    # Decoder
    proj = get_layer(options['decoder'])[1](tparams, Wemb, init_state, options,
                                            prefix='decoder',
                                            mask=mask)

    # Compute word probabilities
    if options['doutput']:
        hid = get_layer('ff')[1](tparams, proj[0], options, prefix='ff_hid', activ='tanh')
        logit = get_layer('ff')[1](tparams, hid, options, prefix='ff_logit', activ='linear')
    else:
        logit = get_layer('ff')[1](tparams, proj[0], options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # Cost
    p_flat = probs.flatten()
    cost = -tensor.log(p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * mask).sum(0)
    cost = cost.sum()

    return trng, [x, mask, ctx, c_idc], cost

def build_sampler(tparams, options, trng):
    """
    Forward sampling
    """
    # the encoded source (sentence, image, etc.): #samples x dim_ctx
    ctx = tensor.matrix('ctx', dtype='float32')

    # cluster indices: 1 x #samples
    c_idc = tensor.vector('c_idc', dtype='int64')

    # Index into the cluster embedding matrix: #sample x dim_char
    n_samples = ctx.shape[0]
    Cemb = tparams['Cemb'][c_idc].reshape([n_samples, options['dim_char']])

    print 'Building f_init...',
    preinit_state = get_layer('ff')[1](tparams, Cemb, options, prefix='pre_ff_state', activ='tanh')
    init_state = get_layer('ff')[1](tparams, ctx + preinit_state, options, prefix='ff_state', activ='tanh')
    f_init = theano.function([ctx, c_idc], init_state, name='f_init', profile=False)

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][y])

    # decoder
    proj = get_layer(options['decoder'])[1](tparams, emb, init_state, options,
                                            prefix='decoder',
                                            mask=None,
                                            one_step=True)
    next_state = proj[0]

    # output
    if options['doutput']:
        hid = get_layer('ff')[1](tparams, next_state, options, prefix='ff_hid', activ='tanh')
        logit = get_layer('ff')[1](tparams, hid, options, prefix='ff_logit', activ='linear')
    else:
        logit = get_layer('ff')[1](tparams, next_state, options, prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [y, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=False)
    print 'Done'

    return f_init, f_next


