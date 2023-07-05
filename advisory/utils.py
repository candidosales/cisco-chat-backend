from advisory.models import Advisory


def mount_advisory_content(data: Advisory):
    return (
        data.id
        + " "
        + data.title
        + " "
        + data.severity
        + " "
        + data.firstPublished
        + " "
        + data.summary
        + " "
        + data.affectedProducts
        + " "
        + data.details
        + " "
        + data.workarounds
        + " "
        + data.workarounds
        + " "
        + data.fixedSoftware
        + " "
        + data.exploitationPublicAnnouncements
        + " "
        + data.source
    )
