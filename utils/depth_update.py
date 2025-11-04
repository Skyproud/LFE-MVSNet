import torch
from utils import binary_tree
import torch.nn.functional as F


def sliding_error_tolerant_fragment(curr_tree_depth, gt_depth_img, b_tree, depth_start, depth_end, is_first):
    with torch.no_grad():
        if depth_start.dim() == 1 or depth_end.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1) #[1,1,1]
            # print(depth_start.shape)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        bin_edge_list = []
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1
        if is_first:
            bin_edge_list.append(torch.zeros_like(gt_depth_img) + depth_start)
            depth_range = depth_end - depth_start
            interval = depth_range / 16.0
            for i in range(16):
                bin_edge_list.append(bin_edge_list[0] + interval * (i + 1))
            gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1
            for i in range(16):
                bin_mask = torch.ge(gt_depth_img, bin_edge_list[i])
                bin_mask = torch.logical_and(bin_mask,
                                             torch.lt(gt_depth_img, bin_edge_list[i + 1]))
                gt_label[bin_mask] = i
            modified_label = gt_label.clone()
            bin_mask = (gt_label != -1)
            return modified_label, gt_label, bin_mask
        if curr_tree_depth == 2:
            depth_range = depth_end - depth_start
            curr_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3))
            curr_interval = depth_range / curr_interval_num
            bin_edge_list = []

            for i in range(17):
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 7, 0)
                tmp_key = torch.minimum(tmp_key, curr_interval_num + 1)
                bin_edge_list.append(curr_interval * tmp_key + depth_start)

            gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1
            for i in range(16):
                bin_mask = torch.ge(gt_depth_img, bin_edge_list[i])
                bin_mask = torch.logical_and(bin_mask,
                    torch.lt(gt_depth_img, bin_edge_list[i + 1]))
                gt_label[bin_mask] = i
            modified_label = gt_label.clone()
            bin_mask = (gt_label != -1)
        elif curr_tree_depth in {3, 4}:
            depth_range = depth_end - depth_start
            curr_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3))
            next_interval = depth_range / curr_interval_num
            bin_edge_list = []

            for i in range(9):
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 3, 0)
                tmp_key = torch.minimum(tmp_key, curr_interval_num + 1)
                bin_edge_list.append(next_interval * tmp_key + depth_start)

            gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1
            for i in range(8):
                bin_mask = torch.ge(gt_depth_img, bin_edge_list[i])
                bin_mask = torch.logical_and(bin_mask,
                    torch.lt(gt_depth_img, bin_edge_list[i + 1]))
                gt_label[bin_mask] = i
            modified_label = gt_label.clone()
            bin_mask = (gt_label != -1)
        elif curr_tree_depth in {5, 6}:
            depth_range = depth_end - depth_start
            curr_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3))
            next_interval = depth_range / curr_interval_num
            bin_edge_list = []

            for i in range(7):
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                tmp_key = torch.minimum(tmp_key, curr_interval_num + 1)
                bin_edge_list.append(next_interval * tmp_key + depth_start)

            gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1
            for i in range(6):
                bin_mask = torch.ge(gt_depth_img, bin_edge_list[i]) # bool
                bin_mask = torch.logical_and(bin_mask,
                    torch.lt(gt_depth_img, bin_edge_list[i + 1])) # bool
                gt_label[bin_mask] = i
            modified_label = gt_label.clone()
            bin_mask = (gt_label != -1)
        else:
            depth_range = depth_end - depth_start
            next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3))
            next_interval = depth_range / next_interval_num # fragment width
            bin_edge_list = []

            ############
            bin_edge_list.append(depth_start - next_interval)
            ############

            for i in range(5):
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key = torch.minimum(tmp_key, next_interval_num + 1)
                bin_edge_list.append(next_interval * tmp_key + depth_start)

            ############
            bin_edge_list.append(bin_edge_list[4] + next_interval)
            ############

            for i in range(6):
                bin_mask = torch.ge(gt_depth_img, bin_edge_list[i])  # bool
                bin_mask = torch.logical_and(bin_mask,
                                             torch.lt(gt_depth_img, bin_edge_list[i + 1]))  # bool
                gt_label[bin_mask] = i

            modified_label = gt_label.clone()
            # update gt_label value
            modified_label = torch.where((modified_label >= 1) & (modified_label <= 4), modified_label - 1, modified_label)  # (-1,0,1,2,3,4)
            modified_label = torch.where(modified_label == 5, modified_label - 2, modified_label)  # (-2,-1,0,1,2,3)

            # Bin_mask that only retains the parts of gt_1abel 0, 1, 2, and 3 as True
            bin_mask = torch.logical_or(torch.eq(modified_label, 0), torch.eq(modified_label, 1))
            bin_mask = torch.logical_or(bin_mask, torch.eq(modified_label, 2))
            bin_mask = torch.logical_or(bin_mask, torch.eq(modified_label, 3))
        return modified_label, gt_label, bin_mask





def update_4pred_4sample(gt_label, curr_tree_depth, b_tree, pred_label, depth_start, depth_end, is_first, with_grad=False, no_detach=False):
    if not with_grad:
        with torch.no_grad():
            # 0 1 2 3
            # -1 0 1 2
            # print(depth_start.dim())
            indicator = torch.ones_like(pred_label) # (1,1,64,80)
            direction1 = pred_label - 7
            direction2 = pred_label - 3
            direction3 = pred_label - 2
            direction4 = pred_label - 1
            if is_first:
                # indicator = torch.zeros_like(pred_label)
                direction = pred_label
            # b_tree:(B,2,H,W)
            # b_tree = binary_tree.update_tree1(b_tree, indicator, direction)
            # print(b_tree)
            
            if depth_start.dim() == 3:
                if depth_start.shape[1] != pred_label.shape[1] or pred_label.shape[2] != pred_label.shape[2]:
                    depth_start = torch.unsqueeze(depth_start, 1)
                    depth_start = F.interpolate(depth_start, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                    depth_start = torch.squeeze(depth_start, 1)

                    depth_end = torch.unsqueeze(depth_end, 1)
                    depth_end = F.interpolate(depth_end, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                    depth_end = torch.squeeze(depth_end, 1)
            elif depth_start.dim() == 1:
                depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
                depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
            
            depth_range = depth_end - depth_start

            if curr_tree_depth  == 1:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0))
                # print(next_interval_num)
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                for i in range(16):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 7, 0)
                    # print(b_tree[:, 1, :, :].shape)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 6
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)
                # print(curr_depth.shape)
            elif curr_tree_depth == 2:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction1)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0)) # 32,64,128,256
                # print(next_interval_num)
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                for i in range(8):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 3, 0)
                    # print(b_tree[:, 1, :, :].shape)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 2
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            elif curr_tree_depth == 3:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction2)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0)) # 32,64,128,256
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                for i in range(8):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 3, 0)
                    # print(b_tree[:, 1, :, :].shape)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 2
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            elif curr_tree_depth == 4:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction2)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0)) # 32,64,128,256
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                for i in range(6):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                    # print(b_tree[:, 1, :, :].shape)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 1
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            elif curr_tree_depth == 5:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction3)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0))
                # print(next_interval_num)
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                for i in range(6):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                    # print(b_tree[:, 1, :, :].shape)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 1
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            elif curr_tree_depth == 6:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction3)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0))
                # print(next_interval_num)
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                for i in range(4):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                    # print(b_tree[:, 1, :, :].shape)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            else:
                b_tree = binary_tree.update_tree1(b_tree, indicator, direction4)
                next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0))
                # print(next_interval_num)
                next_interval = depth_range / next_interval_num
                depthmap_list = []
                tmp_key0 = torch.zeros_like(b_tree[:, 1, :, :])
                tmp_key1 = torch.zeros_like(b_tree[:, 1, :, :])
                slide_tree = b_tree[:, 1, :, :]

                for i in range(4):
                    tmp_key0[gt_label == 0] = torch.clamp_min((slide_tree[gt_label == 0] * 2.0 + i - 2).long(), 0)
                    tmp_key1 = tmp_key1.long()  # if tmp_key1 is floats, then we need long
                    tmp_key1[gt_label == 0] = (slide_tree[gt_label == 0] * 2.0 + i - 1).long()
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    # print(tmp_key1.shape) # [1, 64, 80]

                    condition = (gt_label != 1) | (gt_label != 5)
                    tmp_key0[condition] = torch.clamp_min((slide_tree[condition] * 2.0 + i - 1).long(), 0)
                    tmp_key1 = tmp_key1.long()  # if tmp_key1 is floats, then we need long
                    tmp_key1[condition] = (slide_tree[condition] * 2.0 + i).long()
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

                    tmp_key0[gt_label == 5] = torch.clamp_min((slide_tree[gt_label == 5] * 2.0 + i).long(), 0)
                    tmp_key1 = tmp_key1.long()  # if tmp_key1 is floats, then we need long
                    tmp_key1[gt_label == 5] = (slide_tree[gt_label == 5] * 2.0 + i + 1).long()
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)
                # print(curr_depth.shape)

    else:
        # 0 1 2 3
        # -1 0 1 2
        
        indicator = torch.ones_like(pred_label)
        direction = pred_label - 1
        if is_first:
            # indicator = torch.zeros_like(pred_label)
            direction = pred_label
        b_tree = binary_tree.update_tree1(b_tree, indicator, direction)
        
        if depth_start.dim() == 3:
            if depth_start.shape[1] != pred_label.shape[1] or pred_label.shape[2] != pred_label.shape[2]:
                depth_start = torch.unsqueeze(depth_start, 1)
                depth_start = F.interpolate(depth_start, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                depth_start = torch.squeeze(depth_start, 1)

                depth_end = torch.unsqueeze(depth_end, 1)
                depth_end = F.interpolate(depth_end, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                depth_end = torch.squeeze(depth_end, 1)
        elif depth_start.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        
        depth_range = depth_end - depth_start
        next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 3.0))
        next_interval = depth_range / next_interval_num
        depthmap_list = []
        for i in range(4):
            tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
            tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
            tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

            depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        curr_depth = torch.stack(depthmap_list, 1)
    if no_detach:
        return curr_depth, b_tree
    else:
        return curr_depth.detach(), b_tree.detach()




def depthmap2tree(depth_img, tree_depth, depth_start, depth_end, scale_factor=1.0, mode='bilinear', with_grad=False, no_detach=False):
    if not with_grad:
        with torch.no_grad():
            if scale_factor != 1.0:
                depth_img = torch.unsqueeze(depth_img, 1)
                # print(depth_img.shape) # (1,1,64,80)
                depth_img = F.interpolate(depth_img, scale_factor=scale_factor, mode=mode)
                depth_img = torch.squeeze(depth_img, 1)
                # print(depth_img.shape) # [1, 128, 160]
            B, H, W = depth_img.shape
            b_tree = torch.zeros([B, 2, H, W], \
                dtype=torch.int64, device=depth_img.device)
            b_tree[:, 0, :, :] = b_tree[:, 0, :, :] + tree_depth # 3,5,7
            # print(b_tree[:, 0, :, :])

            if depth_start.dim() == 3:
                if depth_start.shape[1] != depth_img.shape[1] or depth_start.shape[2] != depth_img.shape[2]:
                    depth_start = torch.unsqueeze(depth_start, 1)
                    depth_start = F.interpolate(depth_start, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                    depth_start = torch.squeeze(depth_start, 1)

                    depth_end = torch.unsqueeze(depth_end, 1)
                    depth_end = F.interpolate(depth_end, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                    depth_end = torch.squeeze(depth_end, 1)
            elif depth_start.dim() == 1:
                depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
                depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
            
            depth_range = depth_end - depth_start

            d_interval = depth_range / (2.0 ** (tree_depth + 2)) # the width of every fragment
            b_tree[:, 1, :, :] = (torch.floor((depth_img - depth_start) / d_interval)).type(torch.int64)
            b_tree[:, 1, :, :] = torch.clamp(b_tree[:, 1, :, :], min=0, max=2.0 ** (tree_depth + 2))

            next_interval_num = torch.tensor(2.0 ** (tree_depth + 3.0), device=depth_img.device)
            next_interval = depth_range / next_interval_num
            depthmap_list = []


            if tree_depth == 3:
                for i in range(8):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 3, 0)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 2
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            elif tree_depth == 5:
                for i in range(6):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i - 1
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
                curr_depth = torch.stack(depthmap_list, 1)

            else:
                for i in range(4):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                    tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

                curr_depth = torch.stack(depthmap_list, 1)
    else:
        if scale_factor != 1.0:
            depth_img = torch.unsqueeze(depth_img, 1)
            depth_img = F.interpolate(depth_img, scale_factor=scale_factor, mode=mode)
            depth_img = torch.squeeze(depth_img, 1)
        B, H, W = depth_img.shape
        b_tree = torch.zeros([B, 2, H, W], \
            dtype=torch.int64, device=depth_img.device)
        b_tree[:, 0, :, :] = b_tree[:, 0, :, :] + tree_depth

        if depth_start.dim() == 3:
            if depth_start.shape[1] != depth_img.shape[1] or depth_start.shape[2] != depth_img.shape[2]:
                depth_start = torch.unsqueeze(depth_start, 1)
                depth_start = F.interpolate(depth_start, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                depth_start = torch.squeeze(depth_start, 1)

                depth_end = torch.unsqueeze(depth_end, 1)
                depth_end = F.interpolate(depth_end, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                depth_end = torch.squeeze(depth_end, 1)
        elif depth_start.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        
        depth_range = depth_end - depth_start

        d_interval = depth_range / (2.0 ** tree_depth)
        b_tree[:, 1, :, :] = (torch.floor((depth_img - depth_start) / d_interval)).type(torch.int64)
        b_tree[:, 1, :, :] = torch.clamp(b_tree[:, 1, :, :], min=0, max=2 ** tree_depth)

        next_interval_num = torch.tensor(2.0 ** (tree_depth + 3.0), device=depth_img.device)
        next_interval = depth_range / next_interval_num
        depthmap_list = []

        for i in range(4):
            tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
            tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
            tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

            depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        curr_depth = torch.stack(depthmap_list, 1)
    if no_detach:
        return curr_depth, b_tree
    else:
        return curr_depth.detach(), b_tree.detach()
