from cinematch.data import extract_director, extract_names, parse_json_list


def test_parse_json_list_handles_valid_values():
    assert parse_json_list("[{'name': 'Action'}]") == [{"name": "Action"}]


def test_parse_json_list_handles_invalid_values():
    assert parse_json_list(None) == []
    assert parse_json_list("not-json") == []


def test_extract_names_limits_results():
    raw = "[{'name': 'One'}, {'name': 'Two'}, {'name': 'Three'}]"
    assert extract_names(raw, limit=2) == ["One", "Two"]


def test_extract_director():
    raw = "[{'job': 'Writer', 'name': 'A'}, {'job': 'Director', 'name': 'B'}]"
    assert extract_director(raw) == "B"
