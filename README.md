# quantization
BSc Thesis Quantization

## Todo
- [x] Find optimal transfer learning (flatten vs global average pooling) -> global average pooling
- [x] Check if relu dense is needed -> Yes 1024 is optimal
- [x] Find training speed -> results.xslx
- [x] Check whether trainable false has an influence on training speed -> Yes, a lot!
- [x] Find inference speed -> First predict is slow. Further predicts are faster. Typecasting precision did not work
. Speed reduction is less than training reduction
- [ ] Find accuracy and loss for every model with float32 and mixed
- [ ] Determine whether float16 is worth testing to be trained
- [ ] Test with trainable = true
- [x] Inference is limited by cpu
- [x] Test training speed with multiple workers -> 4 or 8 seems to be optimal
- [ ] Test if evaluation is properly using mixed precision