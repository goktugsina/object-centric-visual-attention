"""Image-derived regions and probability scores on a common pixel support."""
from functools import lru_cache
import numpy as np
from scipy.io import loadmat
from attention.data import fixation_pixels


def box_partition(boxes, confidence, width, height, scale=1., keep=None):
    if scale <= 0:raise ValueError('Box scale must be positive')
    boxes=np.asarray(boxes,float).reshape(-1,4);confidence=np.asarray(confidence,float)
    if len(boxes)!=len(confidence):raise ValueError('Box/confidence mismatch')
    if not np.isfinite(boxes).all() or not np.isfinite(confidence).all():raise ValueError('Nonfinite detections')
    keep=np.ones(len(boxes),bool) if keep is None else np.asarray(keep,bool)
    labels=np.zeros((height,width),np.int32)
    # Paint low-confidence boxes first; earlier indices win exact score ties.
    for index in sorted(np.flatnonzero(keep),key=lambda i:(confidence[i],-int(i))):
        x1,y1,x2,y2=boxes[index];cx,cy=(x1+x2)/2,(y1+y2)/2
        halfx,halfy=(x2-x1)*scale/2,(y2-y1)*scale/2
        left=max(0,min(width,int(np.ceil(cx-halfx-.5))))
        right=max(0,min(width,int(np.ceil(cx+halfx-.5))))
        top=max(0,min(height,int(np.ceil(cy-halfy-.5))))
        bottom=max(0,min(height,int(np.ceil(cy+halfy-.5))))
        if right>left and bottom>top:labels[top:bottom,left:right]=int(index)+1
    return labels


def region_features(detection, class_count, scale=1., keep=None):
    width,height=detection['width'],detection['height']
    labels=box_partition(detection['boxes'],detection['scores'],width,height,scale,keep)
    sizes=np.bincount(labels.ravel(),minlength=len(detection['boxes'])+1)
    active=np.flatnonzero(sizes);area=sizes[active]/labels.size
    yy,xx=np.mgrid[:height,:width]
    cx=np.bincount(labels.ravel(),weights=((xx+.5)/width-.5).ravel(),minlength=len(sizes))[active]/sizes[active]
    cy=np.bincount(labels.ravel(),weights=((yy+.5)/height-.5).ravel(),minlength=len(sizes))[active]/sizes[active]
    confidence=np.r_[0.,detection['scores']][active]
    classes=np.r_[0,detection['classes']].astype(int)[active]
    onehot=np.zeros((len(active),class_count))
    foreground=active!=0;onehot[np.flatnonzero(foreground),classes[foreground]]=1
    features=np.column_stack([np.log(area),cx,cy,cx**2,cy**2,active==0,confidence,onehot])
    return labels,active,area,features


def fixation_records(path):
    records={}
    for record in loadmat(path,simplify_cells=True)['fixations']:
        subjects=[]
        for subject in record['subjects']:
            x,y,duration=[np.atleast_1d(subject[k]) for k in ('fix_x','fix_y','fix_duration')]
            keep=np.isfinite(duration)&(duration>=100)
            px,py,_=fixation_pixels(x[keep],y[keep],800,600)
            subjects.append((px,py))
        records[str(record['img'])]=subjects
    return records


def make_scene(detection, subjects, class_count, scale=1., keep=None):
    labels,active,area,features=region_features(detection,class_count,scale,keep)
    per_subject=np.array([np.bincount(labels[y,x],minlength=len(detection['boxes'])+1)[active] for x,y in subjects])
    counts=per_subject.sum(axis=0)
    return {'name':detection['image'],'features':features,'counts':counts,'subject_counts':per_subject,
            'area':area,'active_ids':active,'pixels':labels.size,
            'foreground_area':float(area[active!=0].sum()),
            'foreground_fixation_fraction':float(counts[active!=0].sum()/counts.sum()),
            'active_detections':int((active!=0).sum())}


def pixel_information_gain(q,p,area):
    q,p,area=map(lambda a:np.asarray(a,float),(q,p,area))
    if q.shape!=p.shape or p.shape!=area.shape:raise ValueError('Support mismatch')
    if np.any(p<=0) or np.any(area<=0):raise ValueError('Positive mass and area required')
    if not all(np.isclose(x.sum(),1) for x in (q,p,area)):raise ValueError('Inputs must be distributions')
    return float(q@np.log2(p/area))


def probability_map(labels,active,p):
    sizes=np.bincount(labels.ravel());lookup=np.zeros(len(sizes))
    lookup[active]=np.asarray(p)/sizes[active]
    return lookup[labels]


@lru_cache(maxsize=8)
def center_log_ratio(width,height,sigma):
    yy,xx=np.mgrid[:height,:width]
    z=-(((xx+.5)/width-.5)**2+((yy+.5)/height-.5)**2)/(2*sigma*sigma)
    weight=np.exp(z)
    return (z-np.log(weight.sum())+np.log(width*height))/np.log(2)


def center_information_gain(subjects,width,height,sigma):
    lookup=center_log_ratio(width,height,sigma)
    values=[lookup[y,x] for x,y in subjects]
    return float(np.concatenate(values).mean())
