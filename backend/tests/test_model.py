from sentiment import SentimentModel


def test_predicts_positive_and_negative(tiny_pipeline_path):
    model = SentimentModel.load(tiny_pipeline_path)
    assert model.predict("a wonderful excellent film").sentiment == "positive"
    assert model.predict("a terrible awful film").sentiment == "negative"


def test_probabilities_sum_to_one(tiny_pipeline_path):
    model = SentimentModel.load(tiny_pipeline_path)
    prediction = model.predict("an excellent movie")
    assert set(prediction.probabilities) == {"positive", "negative"}
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-9
    assert prediction.confidence == max(prediction.probabilities.values())


def test_explanation_signs_match_sentiment(tiny_pipeline_path):
    model = SentimentModel.load(tiny_pipeline_path)
    prediction = model.predict("wonderful excellent brilliant")
    assert prediction.top_features, "expected token contributions"
    # The dominant tokens of a positive prediction must push positively.
    assert prediction.top_features[0].weight > 0

    prediction = model.predict("terrible awful boring")
    assert prediction.top_features[0].weight < 0


def test_unknown_vocabulary_yields_no_features(tiny_pipeline_path):
    model = SentimentModel.load(tiny_pipeline_path)
    prediction = model.predict("zzz qqq xxx")
    assert prediction.top_features == []


def test_top_k_limits_features(tiny_pipeline_path):
    model = SentimentModel.load(tiny_pipeline_path)
    prediction = model.predict("a wonderful brilliant film with excellent acting", top_k=2)
    assert len(prediction.top_features) == 2
