# Paylaşılan araştırma planı: uygulama durumu

## Güncelleme, 5 Kasım 2026

Önceki tarihli durum kaydı aşağıda korunuyor. Yeni tamamlananlar:

- DeepGaze IIE aynı 140 OSIE test görüntüsünde değerlendirildi: 2,695 bit IG; otomatik doğrusal nesne modeli 0,501 bit. NSS, AUC-Judd, CC ve KL da hesaplandı. Farklı ön eğitim bütçeleri nedeniyle bu, tek bir mimari bileşenin etkisini izole etmez.
- EMOd'nin 698 açık görüntüsünde koordinat dönüşümü resmi haritalarla doğrulandı. OSIE ile tam ve algısal hash taramasında eşleşme adayı bulunmadı. Kırpılmış/düzenlenmiş örtüşmeler tümüyle dışlanmış değildir. Diğer 321 IAPS görüntüsü mevcut değildir.
- OSIE'de üç ayrı beş kişilik test grubu eğitim ve doğrulamadan dışlandı. İki bölge türünde 12 doğrusal model yeniden seçildi. Bu kohort içi genelleme deneyidir, kişiselleştirme değildir.
- Yeni ölçütler ve EMOd dönüşümü dahil 31 bilimsel sözleşme testi geçti.

EMOd aktarımı 698 görüntünün tamamında bitti: 6.282 görüntü-model kaydı. Tüm aktarım önbelleklerinin hash denetimi, gözlemci ayrımı kontrolleri ve üç görüntüde tahminlerin yeniden hesaplanması geçti. Sonuçlar `saliency_transfer_results.md` içinde; makalenin yeni sayısal alanları bu sonuçlardan üretilir. Önceki altı sayfalık sürüm `output/archive/2026-11-02/` altında korundu.

ResNet+U-Net veya öğrenilen bölge içi yoğunluk, backbone fine-tuning, EMOd üzerinde DeepGaze, destek/sorgu kişiselleştirme ve sıralı bakış tahmini gelecekteki seçeneklerdir. Colab üzerinde çalıştırma, yazar incelemesi ve dışarıya gönderim tamamlanmış sayılmaz.

Yeni Word/PDF makalesi üretildi ve yedi sayfanın tamamı görsel olarak kontrol edildi. 24 kaynak ve üç tablo var. Yerel gönderim metaverisi yeni PDF hash'iyle yenilendi; dışarıya gönderim yapılmadı. Sonuç raporu: [saliency_transfer_results.md](saliency_transfer_results.md).

## 3 Kasım tarihli durum kaydı

3 Kasım 2026. Kaynak: paylaşılan konuşma (private planning conversation omitted). Konuşmanın son araştırma seçimi **A: Object-Centric Attention Prediction**, orta zorluk, Colab idi. Buradaki çalışma bu araştırma planına aittir; Brown başvuru belgeleri bu kod deposunun kapsamı değildir.

## Önceden tamamlananlar

| Plan görevi | Kanıt / gerçek durum |
|---|---|
| Araştırma sorusu, literatür, veri protokolü | `research/protocol.md`, `literature_review.md`, `research_brief_2026-11-02.md` |
| OSIE görüntüleri, maskeler, bakış kayıtları | 700 görüntü; kaynak ve görüntü hash kayıtları; ham veriler Git dışında |
| Fixation-to-object eşleştirmesi ve background | `src/attention/data.py`; 87.662 uygun bakış; çakışma ve koordinat testleri |
| Nesne dikkat hedefleri, temel modeller | `results/primary/`, `drop_first/`, `primary_nearest/` |
| Semantik katkı ve duyarlılık | `results/robustness/`; beş örtüşen görüntü bölmesi; gözlemci ağırlıklandırma ve özellik deneyleri |
| Görüntüden tahmin ve piksel haritası | Dondurulmuş Faster R-CNN + doğrusal nesne seçimi; `results/image_only/` |
| İlk makale | `paper/manuscript.md`, altı sayfalık PDF ve DOCX; yazar değerlendirmesi/yayın hazırlığı ayrı süreç |

## Bu devam aşamasında tamamlananlar

| Görev | Ürün |
|---|---|
| 10 görüntüde maskeler, bakışlar, iki gözlemci ve hedef dağılımları | `scripts/audit_visual_scenes.py`, `results/visual_audit/manifest.json`, yerel `output/visual_audit/` |
| Dondurulmuş ResNet50 görsel temsili | `scripts/extract_visual_features.py`; aynı 7×7 özellik haritasından iki bölge türü için 2048 boyutlu maske havuzlama |
| Görsel, geometrik ve semantik özellik karşılaştırmaları | `scripts/run_relational.py`; visual, visual_geometry, full_self |
| Nesneler arası ilişki modeli | Tam grafik üzerinde tek başlı additive GAT; self-only ve uniform kontroller |
| Eğitim ve ölçülmüş sonuçlar | İki bölge türü × beş model × üç başlangıç tohumu; `results/relational/`, `relational_results.md` |
| Gözlemci farklılıklarına göre değerlendirme | Eğitim görüntülerinden belirlenen JS eşikleri; testte üç anlaşmazlık grubu, eşit gözlemci NLL |
| Çalıştırılabilir neural görüntü tahmini | `scripts/predict_relational.py`; insan anotasyonu kullanmadan bölge olasılıkları ve normalize piksel haritası |
| Colab başlangıç noktası | `notebooks/object_attention_relations.ipynb`; `scripts/prepare_colab_bundle.py` ile hesap bilgisi gerektirmeyen kaynak paketi |

## Plandan bilinçli uyarlamalar

- Konuşmadaki 490/105/105 örneği yerine mevcut 420/140/140 bölmesi korundu. Önceki deneylerle aynı görüntüler karşılaştırılıyor; bu test kümesi artık keşifsel olarak görülmüş bir kümedir.
- Orta seviye başlangıç için ResNet50 donduruldu. Ayrı ROI crop işlemi yerine tüm görüntüyü koruyan maske ağırlıklı havuzlama kullanıldı. ResNet+U-Net henüz uygulanmadı.
- Özellik deneyleri görsel → görsel+geometri → tüm özellikler sırasındadır. Konuşmadaki her sınıf/konum kombinasyonunun ayrı deneyi tamamlanmış sayılmaz.
- İnsan maskesi/anlambilimi ve detector temsilleri ayrı tutuldu. İnsan özellikleri kullanılan sonuçlar görüntüden çalışan bir ürünün başarısı olarak sunulmuyor.
- Temel değerlendirme proper probability scores ile yapıldı. Spearman, NDCG@3 ve MAE eklendi; NSS, AUC-Judd, CC ve yoğunluk KL tamamlanmış değildir.
- Üç başlangıç tohumu farklı optimizasyon başlangıçlarıdır; yeni veri kümeleri veya bağımsız tekrarlar değildir.
- Colab notebook'u hazırlanıp hücre sözdizimi doğrulanır; bu çalışma yerel CPU üzerinde yürütülür. Colab GPU'da fiilen çalıştırılmış olduğu iddia edilmez.

## Sıradaki araştırma kapıları

1. **Güçlü piksel baseline:** veri/pretraining örtüşmesini inceleyerek uygun dış saliency modelini ve ResNet+U-Net debug baseline'ını aynı görüntü/piksel desteğinde değerlendirmek. DeepGaze III geçmiş bakışlara koşulludur; statik modele karşı koşullandırma farkı açıkça kontrol edilmelidir.
2. **EMOd:** resmi kaynak, erişim koşulları, gözlemci koordinatları, nesne anotasyonları ve OSIE ile görüntü örtüşmesini doğrulamak; uyum varsa modeli yeniden ayarlamadan çapraz veri kümesi testi yapmak.
3. **Daha güçlü gözlemci iddiası:** görülmemiş gözlemciler ve ayrı destek/sorgu kayıtları. Mevcut gruplama kişiye özel tahmin değildir.
4. **Model büyütme kararı:** mevcut doğrulama sonuçları ve hesap bütçesi temelinde backbone fine-tuning. Test sonucuna bakıp hiperparametre ayarlama yapılmamalıdır.

Bu devam aşaması, daha geniş planın tamamının bittiği anlamına gelmez. Mevcut makaleyi güncelleme ve dışarıya gönderme kararları, yeni bulguların yazar tarafından değerlendirilmesinden sonra ele alınır.
