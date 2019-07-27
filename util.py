import torch
import numpy as np

def predict_transform(predictions, inp_dim, anchors, num_classes, cuda_val= True):
    batch_size = predictions.size(0)
    stride = inp_dim // predictions.size(2)
    grid_size = inp_dim//stride
    bbox_attrs = 5 + num_classes ## 5+C Class (5 is tx ,ty,tw,ty,objectness score)
    num_anchors = len(anchors)

    predictions = predictions.reshape(batch_size, bbox_attrs*num_anchors,grid_size*grid_size)
    predictions = predictions.transpose(1,2).contiguous()
    predictions = predictions.reshape(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    ##Initial anchor is for the entire input image. But for the feature map the anchor size
    ## will resize
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    ## Apply sigmoid on tx , ty and objectness score
    predictions[:,:,0] = torch.sigmoid(predictions[:,:,0]) #tx
    predictions[:,:,1] = torch.sigmoid(predictions[:,:,1]) #ty
    predictions[:,:,4] = torch.sigmoid(predictions[:,:,4]) #objectness score

    grid = np.arange(grid_size)
    a,b  = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    #print(torch.cuda.is_available())
    cuda_val = False
    if cuda_val:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset,y_offset), dim=1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    predictions[:,:,:2] += x_y_offset

    ##Apply the anchors to the dimensions of the bounding box.
    ##log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if cuda_val:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    predictions[:,:,2:4] = torch.exp(predictions[:,:,2:4])*anchors

    predictions[:,:,5: 5 + num_classes] = torch.sigmoid((predictions[:,:, 5 : 5 + num_classes]))
    predictions[:,:,:4] *= stride

    return predictions

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

#Calculating the IoU
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # Here prediction[:,:,4] is the class objectness score
    coef_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = coef_mask*prediction

    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = box_corner[:,:,0] - box_corner[:,:,2]/2
    box_corner[:,:,1] = box_corner[:,:,1] - box_corner[:,:,3]/2
    box_corner[:,:,2] = box_corner[:,:,0] + box_corner[:,:,2]/2
    box_corner[:,:,3] = box_corner[:,:,1] + box_corner[:,:,3]/2

    prediction[:,:,4] = box_corner[:,:,4]

    batch_size = prediction.shape(0)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (image_pred[:,:5], max_coef, max_conf_score)

        image_pred = torch.cat(seq,1)
        ## The above 2 lines is equivalent to
        ## torch.cat((image_pred[:,:5], max_coef, max_conf_score),dim = 1)
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7    )

        except :
            continue
         #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1]) # -1 index holds the class index

        for cls in img_classes:
            #perform NMS
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1]== cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(clas_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].reshape(-1,7)

            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] = image_pred_class[i+1:] * iou_mask

                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
