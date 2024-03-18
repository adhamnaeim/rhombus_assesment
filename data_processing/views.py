from django.shortcuts import render
from django.http import JsonResponse
import json
from .scripts.infer_data_types import infer_and_convert_data_types


def process_data(request):
    json_data = infer_and_convert_data_types()
    print(json.loads(json_data))
    return JsonResponse(json_data,safe=False)