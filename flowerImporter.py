img_dir = "PATH/TO/flower_photos"

tulips_df = readImages(img_dir + "/tulips").withColumn("label", lit(1))
daisy_df = readImages(img_dir + "/daisy").withColumn("label", lit(0))

tulips_train, tulips_test = tulips_df.randomSplit([0.6, 0.4])
daisy_train, daisy_test = daisy_df.randomSplit([0.6, 0.4])

train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)