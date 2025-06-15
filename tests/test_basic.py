def test_import_zip2zip_compression():
    try:
        import zip2zip_compression
    except ImportError as e:
        assert False, f"Failed to import zip2zip_compression: {e}"
