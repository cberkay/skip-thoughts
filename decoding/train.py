"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy

import datetime
import errno
import os
import pprint
import pwd
import sys
import time
import warnings

import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import defaultdict

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model, build_sampler
from vocab import load_dictionary
from search import gen_sample

# main trainer
def trainer(X, C, stmodel,
            dim_ctx=4800, # vector dimensionality
            dim_char=4096, # character representation vector dimensionality
            dim_word=620, # word vector dimensionality
            dim=1600, # the number of GRU units
            encoder='gru',
            decoder='gru',
            doutput=False,
            max_epochs=5,
            dispFreq=10,
            decay_c=0.,
            grad_clip=5.,
            n_words=40000,
            n_clusters=10,
            maxlen_w=100,
            optimizer='adam',
            lrate=0.01,
            batch_size = 16,
            saveto='/u/rkiros/research/semhash/models/toy.npz',
            dictionary='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl',
            exp_log_name=None,
            embeddings=None,
            saveFreq=1000,
            sampleFreq=100,
            reload_=False
            ):

    # Model options
    model_options = {}
    model_options['dim_ctx'] = dim_ctx
    model_options['dim_char'] = dim_char
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['decoder'] = decoder
    model_options['doutput'] = doutput
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['n_clusters'] = n_clusters
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['embeddings'] = embeddings
    model_options['saveFreq'] = saveFreq
    model_options['sampleFreq'] = sampleFreq
    model_options['reload_'] = reload_

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print model_options

    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    # Deep dashboard logging
    if exp_log_name is not None:
        username = pwd.getpwuid(os.getuid())[0]

        main_catalogue_path = os.path.join('/u/%s/public_html/' % username,
                                           'catalog')
        if os.path.exists(main_catalogue_path):
            with open(main_catalogue_path, 'a') as f:
                f.write('%s\n' % exp_log_name)
        else:
            with open(main_catalogue_path, 'w') as f:
                f.write('id\n')
                f.write('%s\n' % exp_log_name)

        log_dir = os.path.join('/u/%s/public_html/results/' % username,
                               exp_log_name)
        mkdir_p(log_dir)
        with open(os.path.join(log_dir, 'catalog'), 'w') as f:
            f.write('filename,type,name\n')
            f.write('info.txt,plain,experiment info\n')
            f.write('training_cost.csv,csv,overall training cost\n')
            f.write('samples.txt,plain,training samples\n')

        with open(os.path.join(log_dir, 'info.txt'), 'w') as f:
            f.write(pprint.pformat(model_options))

        with open(os.path.join(log_dir, 'training_cost.csv'), 'w') as f:
            f.write('step,time,cost\n')

    # load dictionary
    print 'Loading dictionary...'
    worddict = load_dictionary(dictionary)

    # Load pre-trained embeddings, if applicable
    if embeddings is not None:
        print 'Loading embeddings...'
        with open(embeddings, 'rb') as f:
            embed_map = pkl.load(f)
        dim_word = len(embed_map.values()[0])
        model_options['dim_word'] = dim_word
        preemb = norm_weight(n_words, dim_word)
        pz = defaultdict(lambda : 0)
        for w in embed_map.keys():
            pz[w] = 1
        for w in worddict.keys()[:n_words-2]:
            if pz[w] > 0:
                preemb[worddict[w]] = embed_map[w]
    else:
        preemb = None

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model...'
    params = init_params(model_options, preemb=preemb)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, inps, cost = build_model(tparams, model_options)

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)
    print 'Done'

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)
    print 'Done'

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    # Each sentence in the minibatch have same length (for encoder)
    train_iter = homogeneous_data.HomogeneousData([X,C], batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    ud_times = numpy.zeros(len(train_iter)*max_epochs, dtype=numpy.float32)

    for eidx in xrange(max_epochs):

        if exp_log_name is not None:

            with open(os.path.join(log_dir, 'catalog'), 'a') as f:
                f.write('epoch_%d_cost.csv,csv,training cost during epoch %d\n' % ((eidx + 1), (eidx + 1)))

            with open(os.path.join(log_dir, 'epoch_%d_cost.csv' % (eidx + 1)), 'w') as f:
                f.write('step,time,cost\n')

        n_samples = 0

        print 'Epoch ', eidx

        for x, c in train_iter:
            n_samples += len(x)
            uidx += 1

            x, mask, ctx, c_idc = homogeneous_data.prepare_data(x, c, worddict, stmodel, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, mask, ctx, c_idc)
            f_update(lrate)
            ud = time.time() - ud_start
            ud_times[uidx - 1] = ud

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Duration ', ud, 'Cum. Av. Dur. ', numpy.mean(ud_times[:uidx])

                if exp_log_name is not None:
                    with open(os.path.join(log_dir, 'epoch_%d_cost.csv' % (eidx + 1)), 'a') as f:
                        f.write('%d,%s,%f\n' %\
                                (uidx,
                                 datetime.datetime.utcnow().isoformat(),
                                 cost))

                    with open(os.path.join(log_dir, 'training_cost.csv'), 'a') as f:
                        f.write('%d,%s,%f\n' %\
                                (uidx,
                                 datetime.datetime.utcnow().isoformat(),
                                 cost))

            if saveFreq is not None and numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                params = unzip(tparams)
                numpy.savez(saveto, history_errs=[], **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            if numpy.mod(uidx, sampleFreq) == 0:

                if exp_log_name is not None:
                    f = open(os.path.join(log_dir, 'samples.txt'), 'a')
                    f.write('Epoch %d Update %d\n\n' % ((eidx + 1), uidx))

                x_s = x
                mask_s = mask
                ctx_s = ctx
                c_idc_s = c_idc
                for jj in xrange(numpy.minimum(10, len(ctx_s))):
                    truth = []
                    sampled = []
                    sample, score =\
                        gen_sample(tparams, f_init, f_next,
                                   (ctx_s[jj].reshape(1, model_options['dim_ctx']),
                                   c_idc_s[jj].reshape(1)),
                                   model_options, trng=trng, k=1, maxlen=100,
                                   stochastic=False, use_unk=False)
                    print 'Truth ',jj,': ',
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            word = word_idict[vv]
                        else:
                            word = 'UNK'
                        truth.append(word)
                        print word,
                    print
                    for kk, ss in enumerate([sample[0]]):
                        print 'Sample (', kk,') ', jj, ': ',
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in word_idict:
                                word = word_idict[vv]
                            else:
                                word = 'UNK'
                            print word,
                            sampled.append(word)
                    print

                    f.write('\tTruth %d: %s\n'.encode('utf-8') % (jj, ' '.join(truth)))
                    f.write('\tSample %d: %s\n'.encode('utf-8') % (jj, ' '.join(sampled)))
                    f.write('\n')

                f.write('\n\n')
                f.close()

        print 'Seen %d samples'%n_samples

    if saveFreq is not None:

        print 'Saving...',

        params = unzip(tparams)
        numpy.savez(saveto, history_errs=[], **params)
        pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))

    print 'Done'

if __name__ == '__main__':
    pass


