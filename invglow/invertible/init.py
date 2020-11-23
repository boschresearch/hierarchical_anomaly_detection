import torch as th

def init_all_modules(net, trainloader, n_batches=10, use_y=False,
                     verbose_init=True):
    with th.no_grad():
        if trainloader is not None:
            if hasattr(trainloader, 'shape') or (
                    hasattr(trainloader, '__getitem__') and
                    hasattr(trainloader[0], 'shape')):
                # then it is/ they are tensors!
                if use_y:
                    init_x = trainloader[0]
                    init_y = trainloader[1]
                else:
                    init_x = trainloader
                    assert not use_y
                    init_y = None
            else:
                all_x = []
                all_y = []
                for i_batch, batch, in enumerate(trainloader):
                    x,y = batch[:2]
                    all_x.append(x)
                    all_y.append(y)
                    if i_batch >= n_batches:
                        break

                init_x = th.cat(all_x, dim=0)
                init_x = init_x.cuda()
                if use_y:
                    init_y = th.cat(all_y)
                else:
                    init_y = None

            for m in net.modules():
                if hasattr(m, 'initialize_this_forward'):
                    m.initialize_this_forward = True
                    m.verbose_init = verbose_init

            _ = net(init_x,fixed=dict(y=init_y))
        else:
            for m in net.modules():
                if hasattr(m, 'initialize_this_forward'):
                    m.initialized = True


def prepare_init(model, verbose_init=True):
    for m in model.modules():
        if hasattr(m, 'initialize_this_forward'):
            m.initialize_this_forward = True
            m.verbose_init = verbose_init
