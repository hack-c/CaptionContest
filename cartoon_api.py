import requests


url_base   = "http://api.cartoonbank.com/?"

def make_request(**kwargs):
    """
    pass keyword arguments corresponding to querystring params.
    return results object
    """
    # construct query string 
    querystring = '&'.join(['{k}={v}'.format(k=k, v=v) for k,v in kwargs.iteritems()])

    # make request
    return requests.get(url_base + querystring)
