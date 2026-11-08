"""Generate extension text and figures directly from completed experiments."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from attention.metrics import bootstrap_mean_ci

ROOT=Path(__file__).resolve().parents[1]
read=lambda name:json.loads((ROOT/'results'/name).read_text())
rob=read('robustness/summary.json');vision=read('image_only/summary.json');oracle=read('image_only/oracle_pixel_reference.json')
assert rob['status']==vision['status']=='complete'
first=vision['runs'][0];m=first['models']['detected_class'];contrast=first['contrasts']['class_over_geometry_confidence']
fmt=lambda x:f'{x:.3f}'
interval=lambda x:f'[{x[0]:.3f}, {x[1]:.3f}]'
range_of=lambda values:f'{min(values):.3f} to {max(values):.3f}'
gains=[r['contrasts']['semantics_over_geometry_event']['improvement_bits'] for r in rob['runs']]
observer_gains=[r['contrasts']['semantics_over_geometry_observer']['improvement_bits'] for r in rob['runs']]
class_gains=[r['contrasts']['class_over_geometry_confidence']['improvement_bits'] for r in vision['runs']]
permuted=[r['contrasts']['permuted_semantics_over_geometry_event']['improvement_bits'] for r in rob['runs']]
no_face=[r['contrasts']['without_face_over_geometry_event']['improvement_bits'] for r in rob['runs']]
no_face_text=[r['contrasts']['without_face_text_over_geometry_event']['improvement_bits'] for r in rob['runs']]
omission=rob['semantic_omission'];stress={r['condition']:r for r in vision['perturbations']}
with (ROOT/'results/image_only/per_image_metrics.csv').open() as f:
    rows=[r for r in csv.DictReader(f) if r['seed']=='20261102' and r['model']=='detected_class']
auto={r['image']:float(r['ig_uniform_bits']) for r in rows}
human={r['image']:r['ig_uniform_bits'] for r in oracle['models']['semantics']['per_image']}
assert set(auto)==set(human)
gap=np.array([human[n]-auto[n] for n in sorted(auto)])
gap_result={'human_semantics_minus_detected_class_bits':float(gap.mean()),'paired_ci95':bootstrap_mean_ci(gap),
            'interpretation':'Joint representation difference; mask geometry, classes and semantic attributes differ.'}
(ROOT/'results/image_only/representation_gap.json').write_text(json.dumps(gap_result,indent=2)+'\n')
fields={}
fields['extension_abstract']=(
    'Object-based attention models often rely on human-defined regions and semantic labels, yet deployment requires regions inferred from images. '
    'We investigate this distinction using 87,662 eligible fixations from 700 OSIE scenes. An annotation-conditioned conditional-choice model is compared with a frozen object detector and a learned region-probability model that uses only images at inference. '
    'Evaluation uses 420 training, 140 validation and 140 test images, with five overlapping image splits for exploratory sensitivity analysis. '
    f'Human semantic attributes improve log score over geometry by {range_of(gains)} bits per fixation across splits. Removing face and text features reduces, but does not eliminate, that benefit. '
    f'For detected objects, class information adds {range_of(class_gains)} bits over geometry and detection confidence. '
    f'On the original split, the image-only model achieves {fmt(m["ig_uniform_bits"])} bits of pixel information gain over a uniform image density, compared with {fmt(first["center_ig_bits"])} bits for a Gaussian center prior and {fmt(oracle["models"]["semantics"]["ig_uniform_bits"])} bits for the human-annotation reference. '
    'Controlled feature omission and box perturbations reveal sensitivity to missing semantics and within-region spatial assumptions. The study provides a reproducible robustness analysis and executable image-only baseline; it does not establish a new attention architecture or neural mechanism.'
)
fields['extension_methods']='''### 4.4 Exploratory robustness analysis

After the original pilot, we specified five split seeds from 20261102 through 20261106. The images overlap across splits; these are sensitivity analyses rather than independent replications. Each model selects L2 from 0, 0.000001, 0.00001, 0.0001, 0.001 and 0.01 using validation data. We compare all attributes, removal of face, removal of text, removal of both, and a control that permutes complete semantic vectors among foreground objects within each image. Geometry and background remain unchanged. One permutation per split is a diagnostic rather than a permutation-test p-value.

An equal-observer evaluation averages each nonempty observer's fixation distribution before scoring. Training still uses the original event-weighted target. This checks the estimand but does not hold observers out. A separate inference-time stress test sets each object's full semantic vector to zero with probability 0, 0.25, 0.5, 0.75 or 1, using 20 deterministic repetitions. Parameters remain fixed. Synthetic omission is not an estimate of natural annotation error.

### 4.5 Image-only object probabilities

The image-only baseline uses Torchvision's Faster R-CNN MobileNetV3 Large 320 FPN with frozen COCO_V1 weights [19,20]. The detector supplies boxes, classes and confidence from pixels. We retain scores of at least 0.5 and at most 50 detections, fixed before gaze evaluation. Higher confidence resolves overlapping boxes, with source order breaking ties; background covers all remaining pixels. Pixel centers determine box membership. Images with no detections remain in the study.

Three conditional-choice models use geometry alone, geometry plus detector confidence, or those seven features plus 91 COCO category slots. Unused category slots are zero. COCO categories are not estimates of OSIE's twelve semantic attributes. Training uses fixation outcomes, but inference uses no human object annotations or gaze. The six-value L2 grid and five image splits match the exploratory analysis. The executable predictor returns a normalized pixel probability map for a new image.

Comparisons across partitions require a common event space. We distribute each region probability uniformly over its pixels. For image i, let a_ij be region area divided by image area. Information gain over a uniform pixel distribution is then:

![Equation 4 Pixel information gain over uniform density](figures/eq_ig.png)

This equals log2 of the number of image pixels minus pixel NLL. Positive values indicate improvement over uniform pixels. For fixed regions, optimizing region cross entropy also optimizes pixel likelihood because the area term is constant with respect to model parameters. We also compare an image-independent Gaussian center density evaluated at the same fixation pixels; its bandwidth is selected on validation images. The original human-mask fits are rescored on this pixel support. Their richer features and different geometry prevent interpreting the gap as a pure detector effect.

With the original image-only model held fixed, we scale box width and height by 0.8 or 1.2 around each box center, or remove detections at probabilities 0.25, 0.5, 0.75 and 1. Features and partitions are recomputed without gaze. Deletion uses one fixed random draw with nested subsets across rates. These interventions are controlled stress tests, not measurements of natural detection accuracy; the tested scale factors are not selected as a new model using test results.'''
table=['| Model | IG bits | 95% interval |','|---|---|---|']
for label,value,ci in [
    ('Uniform pixels',0.,[0.,0.]),
    ('Center prior',first['center_ig_bits'],bootstrap_mean_ci([float(r['center_ig_bits']) for r in rows])),
    ('Detected geometry',first['models']['geometry']['ig_uniform_bits'],first['models']['geometry']['ig_uniform_ci95']),
    ('Geometry + confidence',first['models']['geometry_confidence']['ig_uniform_bits'],first['models']['geometry_confidence']['ig_uniform_ci95']),
    ('Detected classes',m['ig_uniform_bits'],m['ig_uniform_ci95']),
    ('Human geometry',oracle['models']['geometry']['ig_uniform_bits'],oracle['models']['geometry']['ci95']),
    ('Human semantics',oracle['models']['semantics']['ig_uniform_bits'],oracle['models']['semantics']['ci95'])]:
    table.append(f'| {label} | {fmt(value)} | {interval(ci)} |')
fields['extension_results']=f'''### 5.4 Repeated splits and semantic controls

With the expanded regularization grid, semantic gains range from {range_of(gains)} bits across the five splits. Equal-observer scoring yields {range_of(observer_gains)} bits. Removing face reduces the gain to {range_of(no_face)} bits; removing both face and text leaves {range_of(no_face_text)} bits. The attributes therefore contribute unequally, and the semantic effect is not exhausted by those two channels. Within-image permutation yields gains from {range_of(permuted)} bits, offering no consistent benefit.

On the original split, inference-time semantic omission increases NLL from {fmt(omission[0]['nll_bits'])} bits with no omission to {fmt(omission[2]['nll_bits'])} at probability 0.5 and {fmt(omission[-1]['nll_bits'])} with complete omission. These are averages over 20 omission draws. Zeroing the semantic inputs of a fitted model is different from training a new geometry-only model.

### 5.5 Image-only prediction and representation gap

The detector produces no retained boxes for {vision['audit']['zero_detection_images']} of 700 images. Predicted foreground covers {100*vision['audit']['mean_foreground_area']:.2f} percent of image area and {100*vision['audit']['mean_foreground_fixation_fraction']:.2f} percent of eligible fixations on average across images. Fixation coverage is not object-detection recall. Table 2 uses the same 140 test scenes and the same fixation pixels for all methods.

Table 2  Shared pixel information gain on the original split

{chr(10).join(table)}

All entries are bits per fixation, averaged equally over images; larger is better. Intervals resample images and condition on fitted models and the observer cohort. Human annotations are privileged inputs. The detector's COCO pretraining is documented, but overlap with OSIE source photographs has not been exhaustively ruled out.

Detected classes improve over geometry and confidence by {fmt(contrast['improvement_bits'])} bits on the original split, with paired interval {interval(contrast['ci95'])}. Across splits the gain ranges from {range_of(class_gains)} bits. The image-only model's original-split difference from the center prior is {fmt(m['ig_over_center_bits'])} bits, with interval {interval(m['ig_over_center_ci95'])}; it does not consistently outperform that reference across splits. Human semantics exceed the image-only model by {fmt(gap.mean())} bits on the shared pixel support, with paired interval {interval(gap_result['paired_ci95'])}. This is a joint representation gap, not evidence that one isolated component causes the difference.

![Figure 3 Semantic gains over geometry in annotated objects (a) and class gains over geometry plus confidence in detected objects (b). Bars show paired 95 percent image-bootstrap intervals. Splits overlap](figures/extension_splits.png)

### 5.6 Sensitivity to predicted region changes

Shrinking box width and height by 20 percent changes original-split IG from {fmt(stress['original']['ig_uniform_bits'])} to {fmt(stress['shrink_20pct']['ig_uniform_bits'])} bits, while expansion changes it to {fmt(stress['expand_20pct']['ig_uniform_bits'])}. Shrinking reduces foreground fixation coverage from {100*stress['original']['mean_foreground_fixation_fraction']:.2f} to {100*stress['shrink_20pct']['mean_foreground_fixation_fraction']:.2f} percent. Thus higher coverage alone does not imply better probability density. The result is consistent with the uniform-within-box assumption spreading mass too broadly, but it does not isolate that explanation from changes in geometry and overlap.

Deleting detections with probabilities 0.25, 0.5 and 0.75 reduces IG to {fmt(stress['drop_25pct']['ig_uniform_bits'])}, {fmt(stress['drop_50pct']['ig_uniform_bits'])} and {fmt(stress['drop_75pct']['ig_uniform_bits'])} bits. Deleting all detections yields the uniform image density and zero IG by construction. These values describe one fixed deletion draw, so they do not estimate variability across all possible failure patterns.'''
fields['extension_limitations']='''The image-only extension changes the main practical conclusion: semantic information is useful within each representation, but strong performance with human annotations does not establish a competitive image-only attention system. The current detected-region model sometimes falls below a simple center prior. Its class features differ from human semantic attributes, and its uniform spatial distribution within boxes is restrictive. The large human-versus-detected gap therefore motivates matched perception and spatial-density comparisons, rather than attributing the entire gap to detector quality.

The expanded analyses still use one small dataset and the same viewer cohort. Repeated splits overlap, intervention settings are exploratory, deletion uses one random draw, and controlled omission is not a realistic corruption model. Pretrained-detector data overlap has not been exhaustively audited. A second dataset, stronger image-only saliency comparator, repeated perturbations, and held-out observer evaluation remain necessary before broad performance or robustness claims.'''
(ROOT/'paper/extension_fields.json').write_text(json.dumps(fields,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(3.48,3.2),layout='constrained')
for ax,runs,keys in [(axes[0],rob['runs'],[('semantics','All attributes'),('without_face_text','No face/text'),('permuted_semantics','Permuted')]),
                    (axes[1],vision['runs'],[('class','Detected classes')])]:
    for offset,(key,label) in enumerate(keys):
        ds=[r['contrasts']['class_over_geometry_confidence' if key=='class' else key+'_over_geometry_event'] for r in runs]
        means=np.array([d['improvement_bits'] for d in ds]);cis=np.array([d['ci95'] for d in ds])
        ax.errorbar(np.arange(1,6)+(offset-(len(keys)-1)/2)*.12,means,yerr=np.array([means-cis[:,0],cis[:,1]-means]),fmt='o',markersize=3,capsize=2,label=label)
    ax.axhline(0,color='gray',ls='--',lw=.7);ax.set_xticks(range(1,6));ax.legend(fontsize=7)
    ax.set_ylabel('Log-score gain (bits)')
axes[0].set_xlabel('(a) Annotated-object split');axes[1].set_xlabel('(b) Detected-object split')
axes[0].set_ylim(-.025,.25);axes[0].legend(fontsize=7,loc='upper center',ncol=3)
fig.savefig(ROOT/'paper/figures/extension_splits.png',dpi=360);fig.savefig(ROOT/'paper/figures/extension_splits.pdf');plt.close(fig)
report='# Robustness and image-only results\n\nCompleted 2 November 2026. Exploratory extension; not a publication-readiness declaration.\n\n'
report+=fields['extension_results'].replace('### ','## ').replace('(figures/extension_splits.png)','(../paper/figures/extension_splits.png)')
report+='\n\n'+fields['extension_limitations']+'\n'
(ROOT/'research/extension_results.md').write_text(report,encoding='utf-8')
print('Generated extension fields, figure, and evidence report')
