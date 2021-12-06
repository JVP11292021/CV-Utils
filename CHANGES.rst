Version 2.1.0
-------------

Unreleased

-   Drop support for Python 3.6. :pr:`4335`
-   Update Click dependency to >= 8.0. :pr:`4008`
-   Remove previously deprecated code. :pr:`4337`

    -   The CLI does not pass ``script_info`` to app factory functions.
    -   ``config.from_json`` is replaced by
        ``config.from_file(name, load=json.load)``.
    -   ``json`` functions no longer take an ``encoding`` parameter.
    -   ``safe_join`` is removed, use ``werkzeug.utils.safe_join``
        instead.
    -   ``total_seconds`` is removed, use ``timedelta.total_seconds``
        instead.
    -   The same blueprint cannot be registered with the same name. Use
        ``name=`` when registering to specify a unique name.

-   Some parameters in ``send_file`` and ``send_from_directory`` were
    renamed in 2.0. The deprecation period for the old names is extended
    to 2.2. Be sure to test with deprecation warnings visible.

    -   ``attachment_filename`` is renamed to ``download_name``.
    -   ``cache_timeout`` is renamed to ``max_age``.
    -   ``add_etags`` is renamed to ``etag``.
    -   ``filename`` is renamed to ``path``.

-   The ``RequestContext.g`` property is deprecated. Use ``g`` directly
    or ``AppContext.g`` instead. :issue:`3898`
-   ``copy_current_request_context`` can decorate async functions.
    :pr:`4303`