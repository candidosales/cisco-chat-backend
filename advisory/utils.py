from advisory.models import Advisory


def mount_advisory_content(data: Advisory):
    return (
        "ID: "
        + data.id
        + " Title:"
        + data.title
        + " Severity:"
        + data.severity
        + " CVE list:"
        + data.cveList
        + " CVS score:"
        + data.cvsScore
        + " First published:"
        + data.firstPublished
        + " Summary:"
        + data.summary
        + " Affected Products:"
        + data.affectedProducts
        + " Details:"
        + data.details
        + " Workarounds:"
        + data.workarounds
        + " Fixed Software:"
        + data.fixedSoftware
        + " Exploitation and Public Announcements: "
        + data.exploitationPublicAnnouncements
        + " Source:"
        + data.source
        + " URL:"
        + data.url
    )
