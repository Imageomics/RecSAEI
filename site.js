const copyButton = document.getElementById('copy-citation');
const copyStatus = document.getElementById('copy-status');
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex').textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(citation);
    } else {
      const field = document.createElement('textarea');
      field.value = citation;
      field.style.position = 'fixed';
      field.style.opacity = '0';
      document.body.append(field);
      field.select();
      const copied = document.execCommand('copy');
      field.remove();
      if (!copied) throw new Error('Clipboard unavailable');
    }
    copyStatus.textContent = 'Citation copied.';
  } catch {
    copyStatus.textContent = 'Use Download BibTeX to save the citation.';
  }
});
