def build_init_pcure_model(args):
    if 'split' in args.model:
        #vitsrf14x2_split3 # 3,6,9
        if 'vitsrf' in args.model or 'vitlsrf' in args.model:
            model_name = args.model.split('.')[0]
            model_name = model_name.split('_')
            i = model_name[0].find('srf')+3
            window_size = model_name[0][i:].split('x')
            window_size = [int(x) for x in window_size]
            split_point = int(model_name[1][5:])
            print('window_size',window_size,'split_point',split_point)
            if 'vitsrf' in args.model:
                vit = create_model('vit_base_patch16_224',global_pool='avg') #the MAE setup
                vitsrf = vit_base_patch16_224_srf(window_size=window_size)
                load_checkpoint(vit,'checkpoints/mae_finetuned_vit_base.pth')
            elif 'vitlsrf' in args.model:
                vit = create_model('vit_large_patch16_224',global_pool='avg') #the MAE setup
                vitsrf = vit_large_patch16_224_srf(window_size=window_size)
                load_checkpoint(vit,'checkpoints/mae_finetuned_vit_large.pth')       
            load_checkpoint(vitsrf,'checkpoints/{}_vanilla.pth.tar'.format(model_name[0])) ######vanilla not masked yet
            vitsrf,_ = split_vit_like(vitsrf,split_point,True)
            _,vit = split_vit_like(vit,split_point)
            vit.num_window = vitsrf.num_window
            vit.num_patch = vitsrf.num_patch
            vit.window_size = vitsrf.window_size
            model = nn.Sequential(vitsrf,vit)          
        elif 'bagnet' in args.model: #bagnet33_split1 # 1,2,3 # not used in the paper 
            model_name = args.model.split('.')[0]
            model_name = model_name.split('_')
            model_func = BAGNET_FUNC[model_name[0]]
            bn = model_func(pretrained=False,avg_pool=False)
            load_checkpoint(bn,'checkpoints/{}_vanilla.pth.tar'.format(model_name[0])) ######vanilla not masked yet
            rn = create_model('resnet50',pretrained=True)
            print('model loaded!!!!!!!!!!')
            split_point = int(model_name[1][5:])
            print('split_point',split_point)
            bn,_ = split_resnet50_like(bn,split_point)
            _,rn = split_resnet50_like(rn,split_point)
            model = nn.Sequential(bn,rn)           
    elif 'bagnet' in args.model:
        if 'bagnet33' in args.model:
            model = bagnet.bagnet33()
        elif "bagnet17" in args.model:
            model = bagnet.bagnet17()
        elif "bagnet45" in args.model:
            model = bagnet.bagnet45()        
        args.num_classes = 1000
        if args.initial_checkpoint:
            load_checkpoint(model,args.initial_checkpoint)
    elif 'mae' in args.model:
        model = create_model('vit_base_patch16_224',global_pool='avg') #the MAE setup
        load_checkpoint(model,'checkpoints/mae_finetuned_vit_base.pth')
    elif  'vitsrf' in args.model or 'vitlsrf' in args.model:
        #vitsrf14x2_vanilla
        model_name = args.model.split('.')[0]
        model_name = model_name.split('_')
        i = model_name[0].find('srf')+3
        window_size = model_name[0][i:].split('x')
        window_size = [int(x) for x in window_size]
        if 'vitsrf' in args.model:
            vitsrf = vit_base_patch16_224_srf(window_size=window_size)
            if 'masked' in args.model:
                load_checkpoint(vitsrf,'checkpoints/{}_vanilla.pth.tar'.format(model_name[0])) ######vanilla not masked yet
            else:
                load_checkpoint(vitsrf,'checkpoints/mae_finetuned_vit_base.pth')
        elif 'vitlsrf' in args.model:
            vitsrf = vit_large_patch16_224_srf(window_size=window_size)
            if 'masked' in args.model:
                load_checkpoint(vitsrf,'checkpoints/{}_vanilla.pth.tar'.format(model_name[0])) ######vanilla not masked yet
            else:
                load_checkpoint(vitsrf,'checkpoints/mae_finetuned_vit_large.pth')            
        model = vitsrf
    return model