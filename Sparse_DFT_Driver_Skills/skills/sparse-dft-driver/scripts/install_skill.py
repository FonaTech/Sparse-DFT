#!/usr/bin/env python3
"""Install a complete skill; explicitly requested migration keeps a verified backup."""
import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]
IGNORE=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store')


def identity(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.DS_Store'}


def install(skills_dir,migrate_from=None,backup_dir=None,clouds=False):
    root=Path(skills_dir).expanduser().resolve();target=root/SOURCE.name
    if target.exists():raise ValueError(f'Target exists; keep/archive it explicitly before installing: {target}')
    old=None;backup=None
    if migrate_from:
        if Path(migrate_from).name!=migrate_from or migrate_from==SOURCE.name:raise ValueError('migrate-from must be a different simple skill directory name')
        old=root/migrate_from
        if not old.is_dir() or old.is_symlink():raise ValueError('Migration source must be an existing actual skill directory')
        if backup_dir is None:raise ValueError('Migration requires an explicit --backup-dir outside skills-dir')
        backup=Path(backup_dir).expanduser().resolve()/migrate_from
        if backup.is_relative_to(root) or backup.exists():raise ValueError('Choose a new backup directory outside skill discovery roots')
    root.mkdir(parents=True,exist_ok=True)
    staged=root/('.'+SOURCE.name+'-install-'+uuid.uuid4().hex)
    try:
        shutil.copytree(SOURCE,staged,ignore=IGNORE)
        if identity(staged)!=identity(SOURCE):raise ValueError('Installed copy differs from source')
        if clouds:
            text=(staged/'SKILL.md').read_text();body=text.split('---',2)[2].lstrip('\n')
            overlay=json.loads((staged/'agents/clouds-coder.json').read_text())
            (staged/'SKILL.md').write_text('---\n'+json.dumps(overlay,ensure_ascii=False,indent=2)+'\n---\n\n'+body)
        if old:
            before=identity(old);backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(old,backup,ignore=IGNORE)
            if identity(backup)!=before or identity(old)!=before:raise ValueError('Migration source changed during backup; leave old skill active')
        staged.rename(target)
        if old:
            if identity(old)!=before:raise ValueError('Old skill changed during migration; new copy installed but old left active for review')
            shutil.rmtree(old)
        report={'status':'INSTALLED','skill':str(target),'files':len(identity(target)),'retired_skill_backup':str(backup) if backup else None,'available':'next turn'}
        return report
    finally:
        if staged.exists():shutil.rmtree(staged)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skills-dir',type=Path,required=True);p.add_argument('--clouds',action='store_true');p.add_argument('--migrate-from');p.add_argument('--backup-dir',type=Path)
    a=p.parse_args()
    try:print(json.dumps(install(a.skills_dir,a.migrate_from,a.backup_dir,a.clouds),indent=2));return 0
    except (OSError,ValueError) as e:print(json.dumps({'status':'FAILED','message':str(e)}));return 2


if __name__=='__main__':raise SystemExit(main())
