
import torch
from tqdm.auto import tqdm
from opt import config_parser
from sklearn.decomposition import PCA
import os

from renderer import *
from funcs import *

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import datetime
from torchvision import transforms

from dataLoader import dataset_dict
import sys
from pathlib import Path
from third_party import clip
from third import clip as sclip
from einops import repeat



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


def clip_normalize(image):
    image = F.interpolate(image, size=args.patch_size, mode='bilinear')

    b, *_ = image.shape
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    mean = repeat(mean.view(1, -1, 1, 1), '1 ... -> b ...', b=b)
    std = repeat(std.view(1, -1, 1, 1), '1 ... -> b ...', b=b)

    image = (image - mean) / std
    return image

# def ema_update(student, teacher, momentum):
#     # EMA update for the teacher
#     with torch.no_grad():
#         for param_q, param_k in zip(teacher.parameters(), student.parameters()):
#             param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=True)
    test_dataset.read_classes_names()
    if args.has_segmentation_maps:
        test_dataset.read_segmentation_maps()
    c2ws = test_dataset.render_path
    classes = test_dataset.classes
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.get_embed_dim(args.embed_dim)
    tensorf.change_to_feature_mode(device)
    tensorf.load(ckpt)
    tensorf.eval()

    logfolder = os.path.dirname(args.ckpt)

    if args.render_seg_test:
        if args.has_segmentation_maps:
            os.makedirs(f'{logfolder}/segmentations_test', exist_ok=True)
            evaluation_segmentation_test(test_dataset, tensorf, renderer, f'{logfolder}/segmentations_test/',
                                        N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)
        else:
            raise ValueError('the dataset does not have segmentation maps!!')
        
    if args.render_seg_path:
        os.makedirs(f'{logfolder}/segmentations_path', exist_ok=True)
        evaluation_segmentation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/segmentations_path/',
                                        N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)

    if args.render_seg_train:
        os.makedirs(f'{logfolder}/segmentations_train', exist_ok=True)
        evaluation_segmentation_train(test_dataset, tensorf, renderer, f'{logfolder}/segmentations_train/',
                                        N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)

    if args.render_seg_depth:
        os.makedirs(f'{logfolder}/segmentations_depth', exist_ok=True)
        evaluation_segmentation_depth(test_dataset, tensorf, renderer, f'{logfolder}/segmentations_depth/',
                                        N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)

    if args.render_feature:
        text = args.reference_text
        if text is None:
            os.makedirs(f'{logfolder}/feature_pca', exist_ok=True)
            evaluation_feature_pca_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/feature_pca/',
                                        N_vis=-1, N_samples=-1, ndc_ray=ndc_ray,device=device)
        else:
            os.makedirs(f'{logfolder}/feature_text_{text}', exist_ok=True)
            evaluation_feature_text_activation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/feature_text_{text}/',
                                        N_vis=-1, N_samples=-1, ndc_ray=ndc_ray, text=text, device=device)

    if args.render_select:
        os.makedirs(f'{logfolder}/select', exist_ok=True)
        evaluation_select_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/select/',
                                    N_vis=-1, N_samples=-1, ndc_ray=ndc_ray,device=device)


def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    feature_train_dataset = dataset(args.datadir, split='train', patch_size=args.patch_size, downsample=args.ray_downsample_train, is_stack=True, clip_input=args.clip_input)
    feature_train_dataset.read_classes_names()
    if args.has_segmentation_maps:
        feature_train_dataset.read_segmentation_maps()
    patch_train_dataset = dataset(args.datadir, split='train', patch_size=args.patch_size, downsample=args.patch_downsample_train, is_stack=True)
    patch_train_loader = iter(torch.utils.data.DataLoader(patch_train_dataset, batch_size=args.patch_num, sampler=InfiniteSamplerWrapper(patch_train_dataset), num_workers=0, pin_memory=True))
    
    ndc_ray = args.ndc_ray

    # get text features
    classes = feature_train_dataset.classes
    aug_classes = [' '.join([item] * 2) for item in classes]
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
    sclip_model, _ = sclip.load("ViT-B/16", device=device, jit=False)
    text = clip.tokenize(classes).to(device)
    aug_text = clip.tokenize(aug_classes).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text).float() # [N, 512]
        text_features = F.normalize(text_features, dim=1)
        aug_text_features = clip_model.encode_text(aug_text).float() # [N, 512]
        aug_text_features = F.normalize(aug_text_features, dim=1)
    # del clip_model

    # init log file
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    init_logger(Path(logfolder))
    logger.info(args)
    logger.info(f'classes: {classes}')

    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)

    # get pre-computed image CLIP features
    # feature_train_dataset.read_clip_features_and_relevancy_maps(args.feature_dir, text_features, args.test_prompt)

    # init parameters
    aabb = feature_train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    # load pre-trained nerf
    assert args.ckpt is not None, 'Have to be pre-trained to get the density field!'

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.get_embed_dim(args.embed_dim)
    tensorf.load(ckpt)
   
    tensorf.change_to_feature_mode(device)

    # training option
    # out_rgb = False
    dc = DistinctColors()

    # set optimizer
    grad_vars = tensorf.get_optparam_groups_feature_mod(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    # training loop
    torch.cuda.empty_cache()
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout, bar_format='{l_bar}{r_bar}')
    for iteration in pbar:

        if iteration == args.joint_start_iter:
            # change to feature and rgb joint training mode
            out_rgb = True
            tensorf.change_to_feature_rgb_mode()
            tensorf.get_embed_dim(args.embed_dim)
            # new optimizer
            grad_vars = tensorf.get_optparam_groups_feature_rgb_mode(args.lr_init, args.lr_basis)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

        batch = next(patch_train_loader)
        rays_train, rgbs_train, rgbs_d_train = batch['rays'], batch['rgbs'].to(device), batch['rgbs_d'].to(device) #[B, H//8, W//8, 6], [B, H, W, 3]

        with torch.no_grad():
            rgbs = clip_normalize(rgbs_train.permute(0,3,1,2))
            if args.use_maskclip:
                features = clip_model.encode_image(rgbs, dense=True)
            else:
                features = sclip_model.encode_image(rgbs, return_all=True, csa=True)
            rgbs_features = features[:,1:,:]
            # cls = features[:,:1,:]

        feature_shape_wo_dim = rays_train.shape[:3]
        rays_train = rays_train.reshape(-1, 6)
        rgbs_d_train = rgbs_d_train.reshape(-1, 3)

        feature_map, select_map, rgb_map = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples,
                            ndc_ray=ndc_ray, is_train=True, render_feature=True, out_rgb=True, device=device)
        
        rgb_loss = F.mse_loss(rgb_map, rgbs_d_train)

        selected_features = torch.mm(select_map, text_features.T)
        selected_features_T = torch.mm(text_features, select_map.T)

        org_rgbs_features, rgbs_features= tensorf.adapter(rgbs_features.reshape(-1, 512).float())

        if iteration > 8000:
            org_feature_map, feature_map = tensorf.adapter(feature_map)
            self_feature_loss = - (0.3 * (F.cosine_similarity(rgbs_features, org_rgbs_features, dim=1).mean() + \
                                   F.cosine_similarity(feature_map, org_feature_map.detach(), dim=1).mean()) / 2 + \
                                   0.7 * (F.cosine_similarity(feature_map, org_rgbs_features, dim=1).mean() + \
                                   F.cosine_similarity(rgbs_features, org_feature_map.detach(), dim=1).mean()) / 2)
        else:
            self_feature_loss = - F.cosine_similarity(rgbs_features, org_rgbs_features, dim=1).mean()

        feature_loss = - F.cosine_similarity(rgbs_features.detach(), feature_map, dim=1).mean()

        rgbs_relevancy_map = torch.mm(rgbs_features.float(), text_features.T)
        rgbs_relevancy_map_T = torch.mm(text_features, rgbs_features.float().T)
        aug_rgbs_relevancy_map = torch.mm(org_rgbs_features.float(), aug_text_features.T)
        aug_rgbs_relevancy_map_T = torch.mm(aug_text_features, org_rgbs_features.float().T)

        aug_rgbs_relevancy_map_min = torch.min(aug_rgbs_relevancy_map, dim=-1, keepdim=True).values
        aug_rgbs_relevancy_map_max = torch.max(aug_rgbs_relevancy_map, dim=-1, keepdim=True).values
        aug_rgbs_relevancy_map = (aug_rgbs_relevancy_map - aug_rgbs_relevancy_map_min) / (aug_rgbs_relevancy_map_max - aug_rgbs_relevancy_map_min)
        aug_rgbs_relevancy_map_T_min = torch.min(aug_rgbs_relevancy_map_T, dim=-1, keepdim=True).values
        aug_rgbs_relevancy_map_T_max = torch.max(aug_rgbs_relevancy_map_T, dim=-1, keepdim=True).values
        aug_rgbs_relevancy_map_T = (aug_rgbs_relevancy_map_T - aug_rgbs_relevancy_map_T_min) / (aug_rgbs_relevancy_map_T_max - aug_rgbs_relevancy_map_T_min)

        aug_loss = - (F.cosine_similarity(rgbs_relevancy_map, aug_rgbs_relevancy_map).mean() + 
                    F.cosine_similarity(rgbs_relevancy_map_T, aug_rgbs_relevancy_map_T).mean()) / 2


        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        rgbs_features = F.normalize(rgbs_features, dim=-1)

        relevancy_map = torch.mm(feature_map_normalized, text_features.T)
        relevancy_map_T = torch.mm(text_features, feature_map_normalized.T)
        rgbs_relevancy_map = torch.mm(rgbs_features.float(), text_features.T)
        rgbs_relevancy_map_T = torch.mm(text_features, rgbs_features.float().T)
        vis_feature = torch.mm((feature_map_normalized + rgbs_features.float()) / 2, text_features.T)
        vis_feature_T = torch.mm(text_features, (feature_map_normalized + rgbs_features.float()).T / 2)

        log_p_class = F.log_softmax(relevancy_map / args.temperature, dim=1) # [N1,N2]
        log_p_class = log_p_class.reshape(*feature_shape_wo_dim, -1)

        log_p_class_T = F.log_softmax(relevancy_map_T / args.temperature, dim=1) # [N1,N2]
        log_p_class_T = log_p_class_T.reshape(*feature_shape_wo_dim, -1)

        log_rgbs_p_class = F.log_softmax(rgbs_relevancy_map / args.temperature, dim=1) # [N1,N2]
        log_rgbs_p_class = log_rgbs_p_class.reshape(*feature_shape_wo_dim, -1)

        log_rgbs_p_class_T = F.log_softmax(rgbs_relevancy_map_T / args.temperature, dim=1) # [N1,N2]
        log_rgbs_p_class_T = log_rgbs_p_class_T.reshape(*feature_shape_wo_dim, -1)

        vis_log_p_class = F.log_softmax(vis_feature / args.temperature, dim=1) # [N1,N2]
        vis_log_p_class = vis_log_p_class.reshape(*feature_shape_wo_dim, -1)

        vis_log_p_class_T = F.log_softmax(vis_feature_T / args.temperature, dim=1) # [N1,N2]
        vis_log_p_class_T = vis_log_p_class_T.reshape(*feature_shape_wo_dim, -1)


        relevancy_loss = (F.cross_entropy(log_p_class, selected_features.reshape(*log_p_class.shape[:3], -1),) +
                           F.cross_entropy(log_p_class_T, selected_features_T.reshape(*log_p_class.shape[:3], -1))) / 2 + \
                          (F.cross_entropy(log_rgbs_p_class, selected_features.reshape(*log_p_class.shape[:3], -1)) +
                           F.cross_entropy(log_rgbs_p_class_T, selected_features_T.reshape(*log_p_class.shape[:3], -1))) / 2 + \
                          (F.cross_entropy(vis_log_p_class, selected_features.reshape(*log_p_class.shape[:3], -1)) +
                           F.cross_entropy(vis_log_p_class_T, selected_features_T.reshape(*log_p_class.shape[:3], -1))) / 2

        loss = rgb_loss + feature_loss + args.relevancy_weight * relevancy_loss + args.self_feature_weight * self_feature_loss + args.aug_loss_weight * aug_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        #############
        ###logging###
        #############
        if iteration % args.progress_refresh_rate == 0:       
            feature_loss = feature_loss.detach().item()
            relevancy_loss = relevancy_loss.detach().item()
            self_feature_loss = self_feature_loss.detach().item()
            aug_loss = aug_loss.detach().item()
            rgb_loss = rgb_loss.detach().item()

            summary_writer.add_scalar('feature_loss', feature_loss, global_step=iteration)
            summary_writer.add_scalar('relevancy_loss', relevancy_loss, global_step=iteration)
            summary_writer.add_scalar('self_feature_loss', self_feature_loss, global_step=iteration)
            summary_writer.add_scalar('aug_loss', aug_loss, global_step=iteration)
            summary_writer.add_scalar('rgb_mse', rgb_loss, global_step=iteration)


            # Print the current values of the losses.
            pbar.set_description(
                f'feature_loss={feature_loss:.2f},'
                + f'relevancy_loss={relevancy_loss:.2f},'
                + f'self_feature_loss={self_feature_loss:.2f},'
                + f'aug_loss={aug_loss:.2f},'
                + f'rgb_loss={rgb_loss:.2f},'
            )
        ###################
        ###visualization###
        ###################
        if iteration % (args.progress_refresh_rate*20) == 0:
            with torch.no_grad():
                
                # visualize segmentation map
                class_index = torch.argmax(log_p_class, dim=-1, keepdim=True).long()[0] # [B, H//8, W//8, 1]
                class_index = class_index.reshape(-1,1).cpu()

                segmentation_map = dc.apply_colors_fast_torch(class_index).reshape(feature_shape_wo_dim[1], feature_shape_wo_dim[2], 3).permute(2, 0, 1) # [3,H,W]
                segmentation_map = F.interpolate(segmentation_map.unsqueeze(0), size=[rgbs_train.size(1),rgbs_train.size(2)])

                class_rgbs_index = torch.argmax(log_rgbs_p_class, dim=-1, keepdim=True).long()[0] # [B, H//8, W//8, 1]
                class_rgbs_index = class_rgbs_index.reshape(-1,1).cpu()

                segmentation_rgbs_map = dc.apply_colors_fast_torch(class_rgbs_index).reshape(feature_shape_wo_dim[1], feature_shape_wo_dim[2], 3).permute(2, 0, 1) # [3,H,W]
                segmentation_rgbs_map = F.interpolate(segmentation_rgbs_map.unsqueeze(0), size=[rgbs_train.size(1),rgbs_train.size(2)])

                class_rgbs_feature_index = torch.argmax(vis_log_p_class, dim=-1, keepdim=True).long()[0] # [B, H//8, W//8, 1]
                class_rgbs_feature_index = class_rgbs_feature_index.reshape(-1,1).cpu()

                segmentation_rgbs_feature_map = dc.apply_colors_fast_torch(class_rgbs_feature_index).reshape(feature_shape_wo_dim[1], feature_shape_wo_dim[2], 3).permute(2, 0, 1) # [3,H,W]
                segmentation_rgbs_feature_map = F.interpolate(segmentation_rgbs_feature_map.unsqueeze(0), size=[rgbs_train.size(1),rgbs_train.size(2)])


                # visualize feature map
                pca = PCA(n_components=3)
                feature_map = feature_map.reshape(*feature_shape_wo_dim, -1) # [B, H//8, W//8, D]
                feature = feature_map[0].reshape(-1, feature_map.size(-1))
                feature = feature.squeeze().detach().cpu().numpy()
                
                component = pca.fit_transform(feature)
                component = component.reshape([feature_map.size(1), feature_map.size(2), 3])
                component = (component - component.min()) / (component.max() - component.min())
                component = torch.from_numpy(component).permute(2,0,1).unsqueeze(0)
                component = F.interpolate(component, size=[rgbs_train.size(1),rgbs_train.size(2)])

                rgbs_features = rgbs_features.float().reshape(*feature_shape_wo_dim, -1)  # [B, H//8, W//8, D]
                rgbs_features = rgbs_features[0].reshape(-1, rgbs_features.size(-1))
                rgbs_features = rgbs_features.squeeze().detach().cpu().numpy()

                clip_component = pca.fit_transform(rgbs_features)
                clip_component = clip_component.reshape([feature_map.size(1), feature_map.size(2), 3])
                clip_component = (clip_component - clip_component.min()) / (clip_component.max() - clip_component.min())
                clip_component = torch.from_numpy(clip_component).permute(2, 0, 1).unsqueeze(0)
                clip_component = F.interpolate(clip_component, size=[rgbs_train.size(1), rgbs_train.size(2)])



                summary_writer.add_image(
                    'rgb-nerf_seg-clip_seg-nerf+clip',
                    make_grid(
                        [
                            rgbs_train.permute(0,3,1,2)[0].cpu(),
                            segmentation_map[0],
                            segmentation_rgbs_map[0],
                            segmentation_rgbs_feature_map[0],
                        ],
                        padding=0
                    ),
                    global_step=iteration
                )
                summary_writer.add_image(
                    'rgb-feature-clip',
                    make_grid(
                        [
                            rgbs_train.permute(0,3,1,2)[0].cpu(),
                            component.squeeze(),
                            clip_component.squeeze(),
                        ],
                        padding=0
                    ),
                    global_step=iteration
                )

        #############
        ###testing###
        #############
        if (args.clip_input == 1.) and args.has_segmentation_maps and (iteration % (args.progress_refresh_rate*300) == 0 or iteration == (args.n_iters-1)):
            with torch.no_grad():
                savePath = f'{logfolder}/imgs_vis'
                os.makedirs(savePath, exist_ok=True)
                IoUs, accuracies = [], []
                for i, frame_idx in tqdm(enumerate(feature_train_dataset.idxes)):
                    gt_seg = feature_train_dataset.seg_maps[i] # [H*W=N1, n_classes]

                    W, H = feature_train_dataset.img_wh

                    rays = feature_train_dataset.all_rays_stack[frame_idx].reshape(-1, 6)

                    feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=nSamples,
                                        ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
                    
                    _, feature_map = tensorf.adapter(feature_map)
                    
                    feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
                    relevancy_map = torch.mm(feature_map_normalized, text_features.T) # [N1,N2]

                    p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
                    class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
                    segmentation_map = vis_seg(dc, class_index, H, W)

                    one_hot = F.one_hot(class_index.long(), num_classes=gt_seg.shape[-1]) # [N1, n_classes]
                    one_hot = one_hot.detach().cpu().numpy().astype(np.int8)
                    IoUs.append(jaccard_score(gt_seg, one_hot, average=None))
                    accuracies.append(accuracy_score(gt_seg, one_hot))
                    
                    if savePath is not None:
                        imageio.imwrite(f'{savePath}/{iteration:05d}_{frame_idx:02d}.png', segmentation_map)

                # write IoUs to log file
                logger.info(f'\n\niteration: {iteration}')
                logger.info(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n')
                for i, iou in enumerate(IoUs):
                    logger.info(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}')
                    logger.info(f'classes iou: {iou}')        


    tensorf.save(f'{logfolder}/{args.expname}.th')

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20230417)
    np.random.seed(20230417)

    args = config_parser()

    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)

