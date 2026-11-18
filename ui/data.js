// Real baseline from reports/model_comparison.csv (student-test), rounded to 4 dp.
const BASELINE = [
["LogisticRegression",.85,.8519,.7926,.4694,.9521,.1820,.9144,.24,.0009],
["GaussianNB",.8013,.7647,.8208,.5456,.8820,.3938,.9079,.05,.0018],
["DecisionTree",.8561,.8588,.9198,.8595,.8420,.0804,.9171,2.30,.0008],
["RandomForest",.8649,.8692,.9272,.8884,.8414,.0471,.9279,17.44,.0317],
["ExtraTrees",.8616,.8622,.9200,.8933,.8117,.0525,.9256,3.20,.0404],
["HistGradientBoosting",.8694,.8738,.9295,.8825,.8554,.0455,.9270,1.11,.0185],
["MLP",.8601,.8737,.9239,.8889,.8291,.0495,.9300,7.99,.0048]
].map(a => ({src:"baseline",model:a[0],cv_top1_mean:a[1],test_top1:a[2],test_macro_f1:a[3],test_precision:a[4],test_recall:a[5],
  test_log_loss:a[6],route_continuity:a[7],train_time_s:a[8],inference_ms_per_obs:a[9]}));
