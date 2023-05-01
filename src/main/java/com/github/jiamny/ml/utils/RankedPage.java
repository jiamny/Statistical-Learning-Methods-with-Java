package com.github.jiamny.ml.utils;

public class RankedPage {
    private String url;
    private int position;
    private int page;
    private int titleLength;
    private int bodyContentLength;
    private boolean queryInTitle;
    private int numberOfHeaders;
    private int numberOfLinks;

    // getter
    public String getUrl() { return url; }
    public int getPosition() { return position; }
    public int getPage() { return page; }
    public int getTitleLength() { return titleLength; }
    public int getBodyContentLength() { return bodyContentLength; }
    public boolean getQueryInTitle() { return queryInTitle; }
    public int getNumberOfHeaders() { return numberOfHeaders; }
    public int getNumberOfLinks() { return numberOfLinks; }

    // setter
    public void setUrl(String nurl) { url=nurl; }
    public void setPosition(int pst) { position=pst; }
    public void setPage(int pg) { page=pg; }
    public void setTitleLength(int tLt) { titleLength=tLt; }
    public void setBodyContentLength(int bcl) { bodyContentLength=bcl; }
    public void setQueryInTitle(boolean qit) { queryInTitle=qit; }
    public void setNumberOfHeaders(int noh) { numberOfHeaders=noh; }
    public void setNumberOfLinks(int nol) { numberOfLinks=nol; }
}
