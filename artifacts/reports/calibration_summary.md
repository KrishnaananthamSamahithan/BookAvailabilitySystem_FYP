# Multiclass Calibration Summary

## Final Selection
- Base model: `catboost`
- Model-selection validation macro F1: `0.6332`
- Model-selection validation accuracy: `0.7781`
- Accepted calibrator: `temperature_scaling`
- Calibration validation log loss: `0.4052`
- Calibration validation ECE macro: `0.0487`

## Test Set Impact
- Accuracy: `0.7338 -> 0.7338`
- Macro F1: `0.6204 -> 0.6204`
- Log loss: `0.5487 -> 0.5313`
- Multiclass Brier: `0.3129 -> 0.3112`
- ECE macro: `0.0634 -> 0.0583`

## Calibrator Comparison
- `temperature_scaling`: accepted; log_loss=`0.40516088577701537`, ece_macro=`0.048731545243984875`, macro_f1=`0.6338154774906868`
- `none`: accepted; log_loss=`0.40920529074342243`, ece_macro=`0.04932524730365487`, macro_f1=`0.6338154774906868`
- `ovr_isotonic`: rejected (degrades_discrimination_vs_uncalibrated: ba_drop=0.0026, f1_drop=0.0680, max_allowed=0.03); log_loss=`0.37215183274169694`, ece_macro=`0.0021699731243476988`, macro_f1=`0.5657764058281859`
- `multinomial_logistic`: rejected (degrades_discrimination_vs_uncalibrated: ba_drop=0.0038, f1_drop=0.0684, max_allowed=0.03); log_loss=`0.373731711813988`, ece_macro=`0.0012364187660239263`, macro_f1=`0.5654125108280191`
- `dirichlet`: rejected (degrades_discrimination_vs_uncalibrated: ba_drop=0.0040, f1_drop=0.0687, max_allowed=0.03); log_loss=`0.3737937886423938`, ece_macro=`0.0017197378322387792`, macro_f1=`0.5651309870797693`

## Hardest-Calibrated Subgroups
- `route=LHR_TYO`: rows=`213`, ece_macro=`0.1357`, macro_f1=`0.5349`, log_loss=`0.7299`
- `airline_code=ET`: rows=`183`, ece_macro=`0.1130`, macro_f1=`0.4538`, log_loss=`1.0188`
- `airline_code=BA/CZ`: rows=`187`, ece_macro=`0.1095`, macro_f1=`0.5919`, log_loss=`0.7086`
- `meta_engine=Skyscanner`: rows=`8425`, ece_macro=`0.1072`, macro_f1=`0.3109`, log_loss=`0.9569`
- `route=LON_CMB`: rows=`166`, ece_macro=`0.1052`, macro_f1=`0.5827`, log_loss=`0.7884`

## Interpretation
- Temperature scaling wins because it improves probability calibration while preserving the class ranking of the base CatBoost model.
- More expressive calibrators reduced log loss further on validation, but they were rejected because they hurt multiclass discrimination beyond the configured guardrail.
- The remaining research bottleneck is class separation for `price_changed`, not probability sharpness alone.
