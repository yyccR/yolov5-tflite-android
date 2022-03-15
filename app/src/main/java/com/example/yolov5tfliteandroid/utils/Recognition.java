package com.example.yolov5tfliteandroid.utils;

import android.graphics.RectF;


public class Recognition {

    /** Display name for the recognition. */
    private Integer labelId;
    private String labelName;
    private Float labelScore;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private Float confidence;


    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
            final int labelId, final String labelName, final Float labelScore,  final Float confidence, final RectF location) {
        this.labelId = labelId;
        this.labelScore = labelScore;
        this.labelName = labelName;
        this.confidence = confidence;
        this.location = location;
    }

    public Integer getLabelId() {
        return labelId;
    }

    public String getLabelName() {
        return labelName;
    }

    public Float getLabelScore() {return labelScore;}

    public Float getConfidence() {
        return confidence;
    }

    public RectF getLocation() {
        return new RectF(location);
    }

    public void setLocation(RectF location) {
        this.location = location;
    }

    public void setLabelName(String labelName) {this.labelName = labelName;}

    public void setLabelId(int labelId) {this.labelId = labelId;}

    public void setLabelScore(Float labelScore) {
        this.labelScore = labelScore;
    }

    public void setConfidence(Float confidence) {
        this.confidence = confidence;
    }

    @Override
    public String toString() {
        String resultString = "";

        resultString += labelId + " ";

        if (labelName != null) {
            resultString += labelName + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }

        if (location != null) {
            resultString += location + " ";
        }

        return resultString.trim();
    }
}
