"""Recover the author's public MASTER dump; verify MEGA MAC and file content.

No login, private credentials, or private panels are used. Cryptographic format
references: meganz/sdk src/crypto/cryptopp.cpp (CTR/CBC-MAC) and src/utils.cpp
(chunk MAC aggregation). This is data transport, not a MASTER implementation.
"""
from pathlib import Path, PurePosixPath
import argparse
import base64
import hashlib
import json
import os
import struct
import time
import zipfile

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from core.research_integrity_v5.cache import identity, sha
from core.research_integrity_v5.resources import measured

C = Path('/data1/stock-mxy/课题/一进二策略')
B = C / 'research/round23_7_26_1_execution/MASTER_public'
R = C / 'result/round26_1/D5/public_data'
FOLDER = 'MS8mUTbL'
# Public decryption key published in the locked author's README, not an account key.
PUBLIC_KEY = 'qeVz3KR1-MyXc_uLPtkvTg'
SOURCE = f'https://mega.nz/folder/{FOLDER}#{PUBLIC_KEY}'
SESSION = requests.Session()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def decode(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def crypt(data, key, mode, decrypt=False):
    context = Cipher(algorithms.AES(key), mode).decryptor() if decrypt else Cipher(algorithms.AES(key), mode).encryptor()
    return context.update(data) + context.finalize()


def xor(left, right):
    assert len(left) == len(right)
    return bytes(a ^ b for a, b in zip(left, right))


def api(command):
    errors = []
    for attempt in range(3):
        try:
            response = SESSION.post('https://g.api.mega.co.nz/cs', params={'id': str(time.time_ns()), 'n': FOLDER}, json=[command], timeout=(10, 45))
            response.raise_for_status()
            value = response.json()[0]
            if isinstance(value, int):
                raise RuntimeError(f'public API error code {value}')
            return value
        except (requests.RequestException, ValueError, RuntimeError) as error:
            # Do not log temporary signed download URLs or request headers.
            errors.append({'attempt': attempt + 1, 'error_type': type(error).__name__, 'api_error': str(error) if isinstance(error, RuntimeError) else None})
            atomic(R / f'API_FAILURE_{command["a"]}_{time.time_ns()}.json', {'action': command['a'], 'source': SOURCE, 'attempts': errors, 'permission_blocker': False})
            time.sleep(1)
    raise RuntimeError(f'Public MEGA action {command["a"]} failed after3 attempts; see retained API_FAILURE records')


def listing():
    B.mkdir(parents=True, exist_ok=True)
    payload = api({'a': 'f', 'c': 1, 'r': 1, 'ca': 1})
    folder_key = decode(PUBLIC_KEY)
    nodes = {}
    for node in payload['f']:
        if not node.get('a'):
            continue
        found = None
        for entry in node.get('k', '').split('/'):
            if ':' not in entry:
                continue
            try:
                key = crypt(decode(entry.split(':', 1)[1]), folder_key, modes.ECB(), decrypt=True)
                attr_key = xor(key[:16], key[16:]) if node['t'] == 0 else key
                plain = crypt(decode(node['a']), attr_key, modes.CBC(bytes(16)), decrypt=True).rstrip(b'\0')
                if plain.startswith(b'MEGA'):
                    attributes = json.loads(plain[4:])
                    found = {**node, 'name': attributes['n'], '_key': key}
                    break
            except (ValueError, KeyError, UnicodeDecodeError):
                continue
        if found is None:
            raise ValueError('Public node attributes could not be decrypted: ' + node['h'])
        nodes[node['h']] = found
    def path(node, seen=()):
        assert node['h'] not in seen, 'cyclic public folder metadata'
        name = node['name'].replace('\\', '_')
        assert name not in ['', '.', '..'] and '/' not in name
        parent = nodes.get(node.get('p'))
        return (path(parent, seen + (node['h'],)) + '/' if parent else '') + name
    for node in nodes.values():
        node['path'] = path(node)
        assert not PurePosixPath(node['path']).is_absolute() and '..' not in PurePosixPath(node['path']).parts
    public = [{'handle': n['h'], 'path': n['path'], 'type': n['t'], 'bytes': n.get('s')} for n in nodes.values()]
    atomic(R / 'AUTHOR_PUBLIC_LISTING.json', {'source': SOURCE, 'retrieved_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'nodes': public, 'raw_listing_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()})
    return nodes


def chunk_sizes(size):
    remaining = size
    i = 1
    while remaining:
        n = min(remaining, min(i, 8) * 131072)
        yield n
        remaining -= n
        i += 1


def decrypt_file(encrypted, output, key):
    assert len(key) == 32
    aes_key = xor(key[:16], key[16:])
    nonce = key[16:24]
    decryptor = Cipher(algorithms.AES(aes_key), modes.CTR(nonce + bytes(8))).decryptor()
    total_mac = bytes(16)
    digest = hashlib.sha256()
    size = encrypted.stat().st_size
    with encrypted.open('rb') as src, output.open('wb') as dest:
        for n in chunk_sizes(size):
            cipher = src.read(n)
            assert len(cipher) == n
            plain = decryptor.update(cipher)
            dest.write(plain)
            digest.update(plain)
            padded = plain + bytes((-len(plain)) % 16)
            chunk_mac = crypt(padded, aes_key, modes.CBC(nonce + nonce))[-16:]
            total_mac = crypt(xor(total_mac, chunk_mac), aes_key, modes.ECB())
        tail = decryptor.finalize()
        assert not tail
    words = struct.unpack('>4I', total_mac)
    mac = struct.pack('>2I', words[0] ^ words[1], words[2] ^ words[3])
    assert mac == key[24:32], 'MEGA full-file MAC mismatch; unverified plaintext retained as partial'
    assert output.stat().st_size == size
    return {'bytes': size, 'sha256': digest.hexdigest(), 'file_MAC_verified': True}


def download(node):
    target = B / 'files' / node['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    contract = {'source': SOURCE, 'handle': node['h'], 'name': node['path'], 'bytes': node['s'], 'expected_file_MAC': node['_key'][24:32].hex(), 'transport_code_sha256': sha(__file__), 'protocol_sha256': sha(R.parent / 'CONTRACT.yaml')}
    ready = target.with_name(target.name + '.COMPLETE.json')
    if ready.exists() and target.exists():
        prior = json.loads(ready.read_text())
        if prior['contract'] == contract and target.stat().st_size == node['s'] and sha(target) == prior['sha256']:
            return {**prior, 'cache_hit': True}
    encrypted = B / 'encrypted' / (node['h'] + '.enc.partial')
    encrypted.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        offset = encrypted.stat().st_size if encrypted.exists() else 0
        if offset == node['s']:
            break
        assert offset < node['s'], 'Overlong encrypted partial; preserve it and diagnose'
        response_meta = api({'a': 'g', 'g': 1, 'n': node['h']})
        url = response_meta['g']
        assert isinstance(url, str) and url.startswith(('https://', 'http://'))
        # Public content only; no authenticated session or credentials forwarded.
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with SESSION.get(url, headers=headers, stream=True, timeout=(10, 45)) as response:
                if response.status_code == 509:
                    raise RuntimeError('PUBLIC_BANDWIDTH_LIMIT_HTTP509')
                response.raise_for_status()
                if offset:
                    assert response.status_code == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'), 'Server did not honor exact resume; no partial overwritten'
                last = time.monotonic()
                with encrypted.open('ab' if offset else 'wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        offset += len(block)
                        assert offset <= node['s']
                        if time.monotonic() - last >= 10:
                            progress = {'file': node['path'], 'downloaded_bytes': offset, 'expected_bytes': node['s'], 'attempt': attempt + 1}
                            atomic(R / 'DOWNLOAD_PROGRESS.json', progress)
                            print('MASTER_PUBLIC_DOWNLOAD', json.dumps(progress), flush=True)
                            last = time.monotonic()
        except (requests.RequestException, RuntimeError) as error:
            atomic(R / f'DOWNLOAD_FAILURE_{node["h"]}_{time.time_ns()}.json', {'path': node['path'], 'attempt': attempt + 1, 'bytes_retained': encrypted.stat().st_size if encrypted.exists() else 0, 'error_type': type(error).__name__, 'reason': str(error) if isinstance(error, RuntimeError) else 'HTTP_OR_NETWORK_ERROR; temporary URL omitted', 'permission_blocker': False})
            continue
    assert encrypted.exists() and encrypted.stat().st_size == node['s'], 'Public encrypted download incomplete after actual attempts'
    partial = target.with_name(target.name + '.partial')
    actual = decrypt_file(encrypted, partial, node['_key'])
    if target.exists():
        assert sha(target) == actual['sha256'], 'Existing plaintext differs; preserve both, never overwrite silently'
        partial.unlink()
    else:
        os.replace(partial, target)
    record = {**actual, 'contract': contract, 'path': str(target), 'cache_hit': False}
    atomic(ready, record)
    return record


def extract_author_archive(record):
    archive = Path(record['path'])
    records = []
    with zipfile.ZipFile(archive) as reader:
        entries = reader.infolist()
        atomic(R / 'AUTHOR_ARCHIVE_LISTING.json', {'archive': identity(archive), 'entries': [{'path': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size, 'CRC32': x.CRC} for x in entries]})
        selected = [x for x in entries if 'csi300' in x.filename.lower() and x.filename.endswith('.pkl') and any(s in x.filename.lower() for s in ['train', 'valid', 'test'])]
        assert len(selected) == 3, 'Inspect real opensource archive; do not substitute another universe'
        for entry in selected:
            relative = PurePosixPath(entry.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert ((entry.external_attr >> 16) & 0o170000) != 0o120000, 'No archive symlinks'
            target = B / 'extracted' / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            ready = target.with_name(target.name + '.COMPLETE.json')
            contract = {'archive_sha256': record['sha256'], 'member': entry.filename, 'bytes': entry.file_size, 'CRC32': entry.CRC}
            if ready.exists() and target.exists():
                prior = json.loads(ready.read_text())
                if prior['contract'] == contract and target.stat().st_size == entry.file_size and sha(target) == prior['sha256']:
                    records.append({**prior, 'cache_hit': True})
                    continue
            partial = target.with_name(target.name + f'.partial.{time.time_ns()}')
            digest = hashlib.sha256()
            with reader.open(entry) as src, partial.open('wb') as dest:
                while block := src.read(1024 * 1024):
                    dest.write(block)
                    digest.update(block)
            # ZipExtFile has checked the full entry CRC when EOF is reached.
            assert partial.stat().st_size == entry.file_size
            if target.exists():
                assert sha(target) == digest.hexdigest(), 'Preserve conflicting existing public dump'
                partial.unlink()
            else:
                os.replace(partial, target)
            item = {'path': str(target), 'bytes': entry.file_size, 'sha256': digest.hexdigest(), 'archive_CRC_verified': True, 'contract': contract, 'cache_hit': False}
            atomic(ready, item)
            records.append(item)
            print('MASTER_PUBLIC_EXTRACTED', json.dumps(item), flush=True)
    return records


def run(list_only=False):
    nodes = listing()
    files = [n for n in nodes.values() if n['t'] == 0]
    print('MASTER_PUBLIC_LISTING', json.dumps([{'path': n['path'], 'bytes': n['s']} for n in files]), flush=True)
    if list_only:
        return
    archives = [n for n in files if n['name'] == 'opensource.zip']
    chosen = archives or [n for n in files if 'csi300' in n['path'].lower() and n['name'].endswith('.pkl') and any(x in n['name'].lower() for x in ['train', 'valid', 'test'])]
    assert len(chosen) == (1 if archives else 3), 'Inspect actual public folder before choosing source; no guessed alternative files'
    records = []
    with measured(R / 'RECOVERY_RESOURCES.json'):
        for node in sorted(chosen, key=lambda n: n['s']):
            item = download(node)
            if archives:
                atomic(R / 'PUBLIC_ARCHIVE_VERIFIED.json', item)
                records.extend(extract_author_archive(item))
            else:
                records.append(item)
            atomic(R / 'RECOVERY_MANIFEST.json', {'source': SOURCE, 'files': records, 'status': 'PARTIAL' if len(records) < 3 else 'PUBLIC_FILES_CONTENT_VERIFIED'})
    print('MASTER_PUBLIC_RECOVERY_COMPLETE', json.dumps(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-only', action='store_true')
    run(parser.parse_args().list_only)
