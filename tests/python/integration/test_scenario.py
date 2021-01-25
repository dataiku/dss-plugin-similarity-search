import pytest
import logging

from dku_plugin_test_utils import dss_scenario


pytestmark = pytest.mark.usefixtures("plugin", "dss_target")


test_kwargs = {
    "user": "user1",
    "project_key": "TEST_SIMILARITYSEARCHPLUGIN",
    "logger": logging.getLogger("dss-plugin-test.similarity-search.test_scenario"),
}


def test_run_faiss_sentence_embeddings(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="TEST_SIMILARITYSEARCH_PLUGIN_FAISS", **test_kwargs)


def test_run_annoy_sentence_embeddings(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="TEST_SIMILARITYSEARCHPLUGIN_ANNOY", **test_kwargs)


def test_run_faiss_image_embeddings(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="TESTSIMILARITYSEARCHPLUGINIMAGESFAISS", **test_kwargs)


def test_run_edgecases(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="TEST_SIMILARITYSEARCHPLUGIN_EDGECASES", **test_kwargs)


def test_run_partitioning(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="TEST_SIMILARITYSEARCHPLUGIN_PARTITIONED", **test_kwargs)
