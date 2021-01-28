# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario


TEST_PROJECT_KEY = "TEST_SIMILARITYSEARCHPLUGIN"

def test_run_faiss_sentence_embeddings(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TEST_SIMILARITYSEARCH_PLUGIN_FAISS")


def test_run_annoy_sentence_embeddings(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TEST_SIMILARITYSEARCHPLUGIN_ANNOY")


def test_run_faiss_image_embeddings(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TESTSIMILARITYSEARCHPLUGINIMAGESFAISS")


def test_run_edgecases(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TEST_SIMILARITYSEARCHPLUGIN_EDGECASES")


def test_run_partitioning(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TEST_SIMILARITYSEARCHPLUGIN_PARTITIONED")
