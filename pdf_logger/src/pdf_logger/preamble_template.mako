\documentclass[11pt,english,twoside]{article}
\usepackage{katdoc}
\usepackage{mathtools}
\usepackage{mathtools}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cleveref}

\usepackage{tablefootnote}
\usepackage{color}
\usepackage{colortbl}

\newcommand{\docClient}{NRF (National Research Foundation)}
\newcommand{\docFacility}{SARAO (South African Radio Astronomy Observatory)}
\newcommand{\docProject}{MeerKAT Extension}
\newcommand{\docType}{Integration Test Report}
\newcommand{\docFunction}{Engineering / Digital Signal Processing}

\newcommand{\docId}{E1200-XXX-XXXX}
\newcommand{\docRevision}{1}
\newcommand{\docClassification}{Commercial in Confidence}
\newcommand{\docDate}{${DATE}}
\newcommand{\docCopyright}{\copyright SARAO and NRF 2021}

\author{${TESTER_NAME}}

%% Format: \addcontributor{Role}{Name}{Designation}{Affiliation}{Date}
\newcommand{\docApproval}{
    \addcontributor{Submitted by}{J.N. Smith}{Digital Engineer}{SARAO}{}
    \addcontributor{Approved by}{T. van Balla}{Functional Manager: DSP}{SARAO}{}
    \addcontributor{Accepted by}{S. Celliers}{MeerKAT Extention System Engineer}{SARAO}{}
    \addcontributor{Accepted by}{T. Abbott}{MeerKAT Programme Manager}{SARAO}{}
}

%% Format: \addchange{Revision}{Date}{ECN Number}{Comments}
\newcommand{\docHistory}{
    \addchange{A}{01 Apr 2021}{N/A}{First release for internal review.}
}

%% Format: \addprogram{Role}{Package}{Version}
%% TODO: this is going to need to be updated properly.
\newcommand{\docSoftware}{
    \addprogram{Text processor}{pdf\LaTeX}{3.14159265-2.6-1.40.20 (TeX Live 2019/Debian)}{}
}

%% Format: \abbrev{ABBREV}{Definition}
\newcommand{\abbreviations}{
    \abbrev{B-Engine}{The DSP software or bitstream responsible for beamforming.}
    \abbrev{F-Engine}{The DSP software or bitstream responsible for channelisation.}
    \abbrev{F-X Correlator}{A correlator architecture where digitiser data is first channelised (F) per-receptor, and then cross-correlation (X) is calculated per-frequency-channel.}
    \abbrev{X-Engine}{The DSP software or bitstream responsible for correlation.}
}

\newcommand{\applicableDocs}{
  \addReferenceDocument{T. van Balla}{Correlator-Beamformer Requirement Specification}{M1200-0000-000, Rev 4}{22 January 2018}{old_requirement_spec}
}

\newcommand{\referenceDocs}{
  \addReferenceDocument{S. Dennehy}{Correlator-Beamformer Design Document}{M1200-0000-003, Rev 1}{2015-02-07}{old_design_doc}
}

\renewcommand{\floatpagefraction}{0.8}

\begin{document}

\title{Integration Test Report}

\makekatdocbeginning
